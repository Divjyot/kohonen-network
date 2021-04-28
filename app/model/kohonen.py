from . import exceptions as kexceptions
from utils import utils as local_utils
from utils.utils import ParallelProcessingTargets as PPTargets
import copy
import numpy as np
import os.path
import timeit
import logging

_logger: logging.Logger = logging.getLogger(__name__)


class Kohonen(object):
    def __init__(
        self,
        shape: tuple,
        max_iterations: int = 100,
        initial_learning_rate: float = 0.1,
        **kwargs,
    ) -> None:
        """
        Initialise this Kohonen grid.

        Parameters
        ----------
        shape : 2-tuple, list
            2-tuple of width (or x) and height (or y) representing dimensions of Kohonen grid.

        max_iterations: int
            Number of iterations for which the model is going to see the training data.

        initial_learning_rate: float
            Set to a default (0.1), the initial learning rate indicates how quickly the model is going
            to start learning. This changes with time as function of iteration number and time constant.

        """
        super().__init__()

        if not shape or len(shape) != 2:
            raise ValueError(
                f"Invalid shape '{shape}' argument passed. Shape must be iterable representing (width, height) grid dimensions."
            )

        if not max_iterations or max_iterations < 1:
            raise ValueError(
                f"Invalid max iteration '{max_iterations}' argument passed. Max iterations must be a valid integer, greater than 0."
            )

        if not initial_learning_rate or initial_learning_rate < np.nextafter(0, 1):
            raise ValueError(
                f"Invalid initial learning rate '{initial_learning_rate}' argument passed. Please provide a positive number."
            )

        # Initialising variables
        # ------------------------

        # Init grid object that persists nodes coordinate positions.
        self.shape_x, self.shape_y = shape
        self.grid = np.array(list(np.ndindex(self.shape_x, self.shape_y)))

        # Hyperparameters
        # ---------------
        self.max_iterations = max_iterations
        self.ini_learning_rate = initial_learning_rate
        self.learning_rate = initial_learning_rate

        neigh_radius = max(shape) / 2
        self.ini_neigh_radius = neigh_radius
        self.curr_neigh_radius = neigh_radius
        self.time_constant = max_iterations / np.log(neigh_radius)

        # Verbose settings
        # ----------------
        self.bmu_indicies_track = []
        self.neigh_radii_track = []
        self.learning_rate_track = []

        # Debug Mode
        # ----------
        # Ideal for testing and printing stats.
        self.debug_mode = kwargs.get("debug_mode", False)

        # Verbose
        # -------
        self.verbose = kwargs.get("verbose", False)

        # For large computation (higher resolution grid),
        # target multiple places that can use multiprocessing.
        # For more details please refer to kohonen.ipynb's Optimisation section.
        self.apply_parallel_processing = {
            PPTargets.FIND_BMU: kwargs.get(PPTargets.FIND_BMU.value, False),
            PPTargets.INF_BMU_W: kwargs.get(PPTargets.INF_BMU_W.value, False),
            PPTargets.INF_BMU_POS: kwargs.get(PPTargets.INF_BMU_POS.value, False),
        }

        if self.debug_mode:
            print(f"Is parallel processing ? : {self.apply_parallel_processing}")

    # Public
    def train(self, input: np.ndarray):
        """
        Trains the grid by following steps.

        1. Each node's weights are initialized.
        2. Enumerate through the training data for some number of iterations (repeating if necessary).
            The current value we are training against will be referred to as the current input vector
            2.1 Every node is examined to calculate which one's weights are most like the input vector.
                The winning node is commonly known as the Best Matching Unit (BMU).
            2.2 The radius of the neighbourhood of the BMU is now calculated.
                This is a value that starts large, typically set to the 'radius' of the lattice,
                but diminishes each time-step. Any nodes found within this radius are deemed to be
                inside the BMU's neighbourhood.
            2.3 Each neighbouring node's (the nodes found in step 2.2) weights are adjusted to make
                them more like the input vector. The closer a node is to the BMU, the more its
                weights get altered.
        3. Go to step 2 until we've completed N iterations.

        Parameters
        ----------
        input : numpy.array
            Numpy array of shape (X, Y) where X indicates total number of training items and
            Y indicates total number of dimensions.
            A training item is numpy array representing features values.

        Returns
        -------
        result : dict
            Dictionary that contains 'message', 'model_name'(if saved) key(s).
            'message' : general training information such as training time.

            if self.verbose is True
            'labels' : list of grid node's indicies that are closest per training-item.
            'learning_rates' : list of all learning rates over time/iterations.
            'neigh_radii' : list of all radii over time/iterations.
            'bmu_indicies' : list of all BMU positions over time/iterations.

        Raises
        ------
        KohonenInputError
            If weights are of size zero or input is of type other than numpy.array
        """
        response = {}
        if isinstance(input, np.ndarray) & (input.size != 0):
            # Init Weights: float values derived by normalising random number in range [0-255)
            self.set_weights(
                np.array(np.random.randint(0, 255, size=(self.shape_x, self.shape_y, input.shape[-1])) / 255)
            )
            overall_start_time = timeit.default_timer()
            # Iterate from 1 to max_iterations
            for iteration in range(1, self.max_iterations + 1):
                itr_start_time = timeit.default_timer()
                # Enumerate through training data; fine BMU, calculate influence and update weights.
                [self._train_on_current_vector(current_input_vector) for current_input_vector in input]
                self._update_BMU_neighbourhood(iteration)
                self._update_learning_rate(iteration)
                if self.debug_mode:
                    print(
                        iteration,
                        ": {:.2f}".format(timeit.default_timer() - itr_start_time),
                        "\n",
                    )
            time_taken = "{:.2f}".format(timeit.default_timer() - overall_start_time)
            message = f"Model training has finished and time taken is {time_taken} seconds."
            print(message)
            if self.verbose:
                predictions = self.predict(input)
                response["predictions"] = predictions.tolist()
                response["learning_rates"] = self.learning_rate_track
                response["neigh_radii"] = self.neigh_radii_track
                response["bmu_indicies"] = self.bmu_indicies_track
        else:
            message = f"Invalid input: '{input}', please provide a numpy array with shape (width, height, depth)"
            _logger.exception(message)
            raise kexceptions.KohonenInputError(message)
        response["message"] = message
        return response

    def predict(self, test_input: np.ndarray):
        """
        Associates each testing-item in param:test_input to the closest node in grid based on
        eucledian distances. This function can be useful to associate a new input item to clusters
        formed by the trained model.

        Parameters
        ----------
        test_input : numpy.array
            Numpy array of shape (X, Y) where X indicates total number of testing items.
            a testing-item (alias current test vector) is numpy array representing features values.

        Returns
        -------
        indicies : list
            Indicies of grid node which is the most closest to testing-item.

        """
        _flatten_weights = self._flatten_weights()
        return np.array(
            [self._find_closest(current_test_vector, _flatten_weights) for current_test_vector in test_input]
        )

    def save(self, model_name: str):
        """
        Ideally these weights can be saved to a external blob storage & retrevied when needed,
        however for the purpose of this assignment, the weights are saved locally under
        'saved_models' folder in <model_name>.npy format.

        Parameters
        ----------
        model_name : str [optional]
            Model weights will be saved to the file.

        Returns
        -------
        model_name : str
            Model name string.

        Raises
        -------
        KohonenMissingWeightsError
            If the model weights have size of zero.
        """
        if self._weights.size != 0:
            model_path = local_utils.get_model_path(model_name)
            np.save(model_path, self._weights)
            return model_name
        raise kexceptions.KohonenMissingWeightsError(f"The model has no weights to save.")

    @classmethod
    def load(cls, model_name: str):
        """
        Loads model weights and sets them to self weights.

        Parameters
        ----------
        model_name : str
            Model weights will be loaded from the weight file named this.

        Returns
        -------
        weights : numpy.array
            Weights is a numpy array of shape (width, height, dimensions).

        Raises
        ------
        FileNotFoundError
            If a file named 'model_name' do not exists.
        """
        model_path = local_utils.get_model_path(model_name)
        if os.path.isfile(model_path):
            return np.load(model_path)
        raise FileNotFoundError(f"No saved model under the name '{model_name}' found.")

    def set_weights(self, weights):
        """
        Helper method to sets weights and dimensions in self object.

        Parameters
        ----------
        weights : numpy.array
            Weights is a numpy array of shape (width, height, dimensions).

        Raises
        -------
        exception : KohonenInputError
            If the model weights have size of zero.

        """
        if weights.size == 0:
            raise kexceptions.KohonenInputError(
                f"Weights of size 0 cannot be set. Please provide correct weights of shape (width, height, dimensions)."
            )

        self._weights = weights
        self.shape_x = self._weights.shape[0]  # grid rows
        self.shape_y = self._weights.shape[1]  # grid columns
        self.shape_z = self._weights.shape[2]  # grid depth or dimensions

    def get_weights(self):
        """
        Gets model weights.

        Raises
        -------
        exception : KohonenMissingWeightsError
            If the model weights have size of zero.
        """
        if self._weights.size != 0:
            return self._weights

        raise kexceptions.KohonenMissingWeightsError(f"Critial error, the model has no weights.")

    # Privates
    def _train_on_current_vector(self, current_input_vector):
        """
        Orchestrates training of grid on individual training-item.

        Parameters
        ----------
        current_input_vector : numpy.array
            Single training-item i.e. a colour.

        """
        # Find BMU
        bmu_xy = self._find_BMU(current_input_vector)

        # Evaluate BMU's influence in the neighbourhood
        influence_vector = self._calculate_influence(bmu_xy)

        # Update weights : function of learning rate, influence vec, current input vec
        self._update_weights(current_input_vector, influence_vector)

    def _find_BMU(self, current_input_vector: np.array):
        """
        Determines Best Matching Unit (BMU).

        Parameters
        ----------
        current_input_vector : numpy.array
            current training value, the model is training against.

        Returns
        -------
        (bmu_x, bmu_y) : 2-tuple
            BMU position's in the grid.

        """
        bmu_x, bmu_y = self._find_closest(current_input_vector, self._flatten_weights())
        if self.verbose:
            self.bmu_indicies_track.append([int(bmu_x), int(bmu_y)])
        return bmu_x, bmu_y

    def _calculate_influence(self, bmu_xy):
        """
        Calculates influence of BMU based on its Pythagorian distance between its position
        and all other nodes positions as function of neighbourhood radius at current iteration.

        Parameters
        ----------

        bmuxy : 2-tuple (int, int)
            Consisting of BMU node's x and y position in the grid.

        Returns
        ----------
        influence_vector : numpy.array
            Array of shape (shape_x, shape_y, 1),
            where shape_x and shape_y represents of grid rows(width) and columns(height)

        """
        # 1. Identify BMU neighbourhood nodes
        bmu_x, bmu_y = bmu_xy
        bmu = self._weights[bmu_x, bmu_y]
        # DEBUG: are we mearsuring time
        start_time = self.debug_mode
        if self.debug_mode:
            start_time = timeit.default_timer()
        if not self.apply_parallel_processing[PPTargets.INF_BMU_W]:
            # using apply_along_axis
            # eucledian_distances = np.apply_along_axis(
            #     local_utils.eucledian_between_vec, 1, self._flatten_weights(), bmu
            # )
            eucledian_distances = local_utils.euc(bmu, self._flatten_weights())
            if self.debug_mode:
                print(
                    f"{PPTargets.INF_BMU_W.value}:Sequential:",
                    "{:.2f}".format(timeit.default_timer() - start_time),
                )
        else:
            # using multiprocessing-apply_along_axis
            eucledian_distances = local_utils.parallel_apply_along_axis(
                local_utils.eucledian_between_vec, 1, self._flatten_weights(), bmu
            )

            if self.debug_mode:
                print(
                    f"{PPTargets.INF_BMU_W.value}:Parallel:",
                    "{:.2f}".format(timeit.default_timer() - start_time),
                )

        # Identify order of each element by arranging index.
        eucledian_distances = np.argsort(np.argsort(eucledian_distances.reshape(self.shape_x * self.shape_y)))

        # 2. Identify nodes that are inside, outside current neighbourhood radius.
        # Setting node-mux vector value to 0 deems the node for pruning in current iteration.
        prune_mux_vector = copy.deepcopy(eucledian_distances)
        prune_mux_vector[eucledian_distances > self.curr_neigh_radius] = -1
        prune_mux_vector[prune_mux_vector == -1] = 0
        prune_mux_vector[prune_mux_vector >= 0] = 1

        # 3. Using Pythagoras, eucledian distances between bmu and every other node
        # minus outside neighbourhood, are calculated.
        if self.debug_mode:
            start_time = timeit.default_timer()
        if not self.apply_parallel_processing[PPTargets.INF_BMU_POS]:
            # using apply_along_axis
            # influence_distances = np.apply_along_axis(local_utils.eucledian_between_point, 1, self.grid, bmu_xy)
            influence_distances = local_utils.euc(np.array(bmu_xy), self.grid)
            if self.debug_mode:
                print(
                    f"{PPTargets.INF_BMU_POS.value}_POS:Sequential:",
                    "{:.2f}".format(timeit.default_timer() - start_time),
                )
        else:
            # using multiprocessing-apply_along_axis
            influence_distances = local_utils.parallel_apply_along_axis(
                local_utils.eucledian_between_vec, 1, self.grid, bmu_xy
            )
            if self.debug_mode:
                print(
                    f"{PPTargets.INF_BMU_POS.value}:Parallel:",
                    "{:.2f}".format(timeit.default_timer() - start_time),
                )

        influence = np.exp(-(np.square(influence_distances) / (2 * (self.curr_neigh_radius ** 2))))

        # Remove influence of nodes outside current radius:
        return np.multiply(influence, prune_mux_vector).reshape(self.shape_x, self.shape_y, 1)

    def _update_weights(self, current_input_vector, influence_vector):
        """
        Updates weights of all nodes based on current input vector and influence vector.

        Parameters
        ----------
        current_input_vector : numpy.array
            current training value, the model is training against.

        influence_vector : numpy.array
            Array of shape (shape_x, shape_y, 1)
        """
        self._weights = self._weights + (
            self.learning_rate * np.multiply(influence_vector, (current_input_vector - self._weights))
        )

    def _update_BMU_neighbourhood(self, iteration):
        """
        Updates current neighbourhood radius based on interation number.

        Parameters
        ----------
        iteration : int
            Current iteration number.
        """
        self.curr_neigh_radius = self.ini_neigh_radius * (np.exp(-iteration / self.time_constant))
        if self.verbose:
            self.neigh_radii_track.append(float(self.curr_neigh_radius))

    def _update_learning_rate(self, iteration):
        """
        Updates current learning rate based on iteration number and time constant.

        Parameters
        ----------
        iteration : int
            Current iteration number.
        """
        self.learning_rate = self.ini_learning_rate * (np.exp(-iteration / self.time_constant))
        if self.verbose:
            self.learning_rate_track.append(float(self.learning_rate))

    def _flatten_weights(self):
        """
        Reshapes/flattens weights from (rows, columns, dimensions) to
        (rows * columns, dimensions)

        Returns
        -------
        weights : numpy.array
            shape of (rows * columns, dimensions)
        """
        if self._weights.size != 0:
            return self._weights.reshape(self.shape_x * self.shape_y, self.shape_z)

        raise kexceptions.KohonenMissingWeightsError(f"Critial error, the model has no weights.")

    def _find_closest(self, current_vector, grid_weight_vecs, grid_wise=True):
        """
        Finds the closest node to param:current_vector in grid based on eucledian distance.

        Parameters
        ----------
        current_vector : numpy.array
            A input-item representing one training-item/testing-item.

        grid_weight_vecs : numpy.array
            Array representing the grid weights of shape (N,M,)

        grid_wise : bool
            Flag to indicate the type of postition to be returned.

        Returns
        -------
        position : int, 2-tuple(int)
            if postition set to True, (x,y):2-tuple coordinates will be returned
            else (x):int will be returned.

            (x) indicates position in a (rows * columns, dimensions) shape of weights.
            (x,y) indicates position in a (rows, columns, dimensions) shape of weights.

        """

        start_time = self.debug_mode
        if self.debug_mode:
            start_time = timeit.default_timer()

        if not self.apply_parallel_processing[PPTargets.FIND_BMU]:
            # eucledian_distances = np.apply_along_axis(
            #     local_utils.eucledian_between_vec,
            #     1,
            #     grid_weight_vecs,
            #     current_vector,
            # )
            eucledian_distances = local_utils.euc(current_vector, grid_weight_vecs)
            if self.debug_mode:
                print(
                    f"{PPTargets.FIND_BMU.value}:Sequential:",
                    "{:.2f}".format(timeit.default_timer() - start_time),
                )
        else:
            eucledian_distances = local_utils.parallel_apply_along_axis(
                local_utils.eucledian_between_vec,
                1,
                grid_weight_vecs,
                current_vector,
            )
            if self.debug_mode:
                print(
                    f"{PPTargets.FIND_BMU.value}:Parallel:",
                    "{:.2f}".format(timeit.default_timer() - start_time),
                )

        position = np.where(eucledian_distances == eucledian_distances.min())
        position = position[0][0]
        if grid_wise:
            eucledian_distances = eucledian_distances.reshape(self.shape_x, self.shape_y)
            position = np.where(eucledian_distances == eucledian_distances.min())
            position = position[0][0], position[1][0]  # (node's x, node's y)
        return position
