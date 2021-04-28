from settings import NPY_EXT, MODELS_DIR

import os
import math
import numpy as np
from datetime import datetime

def euc(vec:np.array, pC:np.array):
    pC_vec = np.full((pC.shape[0], pC.shape[1]),  vec)
    step1 = np.subtract(pC,  pC_vec)
    step2 = np.square(step1)
    step3 = np.sum(step2, axis=1, dtype=float).reshape(pC.shape[0],)
    step4 = np.sqrt(step3, dtype=float)
    return step4

def eucledian_between_point(point1: tuple, point2: tuple):
    """
    Return eucledian distance between two points.

    Parameters
    ----------
    point1 : tuple
        (x,y) coordinate pair.

    point2 : tuple
        (x,y) coordinate pair.

    Returns
    -------
    Eucledian distance between both vectors.
    """
    point1_x, point1_y = point1
    point2_x, point2_y = point2
    return math.sqrt(((point1_x - point2_x) ** 2) + ((point1_y - point2_y) ** 2))


def eucledian_between_vec(vec1: np.array, vec2: np.array):
    """
    Return eucledian distance between two vectors.

    Parameters
    ----------
    vec1 : numpy.array
        Array contains coordinate set of points.

    vec2 : numpy.array
        Array contains coordinate set of points.

    Returns
    -------
    Eucledian distance between both vectors.
    """
    return np.sqrt(np.sum(np.square(np.subtract(vec1, vec2))))


def get_model_path(model_name):
    """
    Returns a path with extension based on param:model_name.

    Parameters
    ----------
    model_name : str
        Name of file under which weights are saved.
    """
    model_name = model_name.replace(NPY_EXT, "")
    return os.path.join(MODELS_DIR, f"{model_name}{NPY_EXT}")


def generate_model_name(grid_size, max_iterations, learning_rate):
    """
    Parameters
    ----------
    grid_size : api_params.TrainParameters
        Same parameter object used for training a model.

    max_iterations : int
        Max iterations that model training on.

    learning_rate : float
        Learning rate that model training on.

    Returns
    -------
    model_name : str
        A unique string build using parameters attributes.
    """
    grid_x, grid_y = grid_size
    return f"{datetime.now().strftime('%d-%m-%Y_%Hh%Mm%Ss')}T_{grid_x}X{grid_y}_{max_iterations}N_{learning_rate}LR"


##############################################################################
import multiprocessing
import enum


class ParallelProcessingTargets(enum.Enum):
    """
    This enum class helps to facilitate boolean flags in code
    to isolate parallel processing code for conditional execution.
    """

    FIND_BMU = "pp_FIND_BMU"
    INF_BMU_W = "pp_INF_BMU_W"
    INF_BMU_POS = "pp_INF_BMU_POS"


def apply_along_axis_wrapper(apply_along_axis_args):
    """
    Wrapper around numpy.apply_along_axis().

    Parameters
    ----------
    apply_along_axis_args : n-tuple
        Tuple containing arguments to numpy.apply_along_axis arguments


    Returns
    -------
    A numpy array to which func1D has applied.
    """
    (func1d, axis, arr, args, kwargs) = apply_along_axis_args
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)


def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    A multiprocessing variant of numpy.apply_along_axis() which divides the
    numpy.array into n-chunks based on the number of CPUs. It processes these
    chunks in parallel and later concates the results from each chunk into a array.


    Parameters
    ----------
    func1d : function
        A function that has to map to numpy array.

    axis : int (0,1)
        Axis along which arr is sliced.

    arr : ndarray (Ni…, M, Nk…)
        Input array

    args : any
        Additional arguments to func1d.

    kwargs : any
        Additional named arguments to func1d.

    Returns
    -------
    A numpy array to which func1D has applied.
    """

    pool = multiprocessing.Pool()
    chunks = [
        (func1d, axis, arr_chunk, args, kwargs) for arr_chunk in np.array_split(arr, multiprocessing.cpu_count())
    ]
    chunk_results = pool.map(apply_along_axis_wrapper, chunks)
    pool.close()
    pool.join()
    return np.concatenate(chunk_results)
