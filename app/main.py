from settings import HTTPCode, NPY_EXT, MODELS_DIR
import utils.utils as local_utils
from model.kohonen import Kohonen
import api_params
import logging
import numpy as np
from os import listdir
from os.path import isfile, join
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from utils.utils import ParallelProcessingTargets as PPTargets

_logger: logging.Logger = logging.getLogger(__name__)
app = FastAPI()


def _train(parameters: api_params.TrainParameters, model_identifier):
    """
    Parameters
    ----------
    parameters : api_params.TrainParameters
        See app.api_parameters.TrainParameters class for detail.

    model_identifier : str
        Unique identifier to name a training run.

    Returns
    -------
    message : dict
    """
    training_data = np.array(parameters.training_data).reshape(parameters.training_data_shape)
    model = Kohonen(
        shape=parameters.grid_shape,
        max_iterations=parameters.max_iterations,
        initial_learning_rate=parameters.learning_rate,
        debug_mode=parameters.debug_mode,
        verbose=parameters.verbose,
        **{
            PPTargets.FIND_BMU.value: parameters.pp_FIND_BMU,
            PPTargets.INF_BMU_W.value: parameters.pp_INF_BMU_W,
            PPTargets.INF_BMU_POS.value: parameters.pp_INF_BMU_POS,
        },
    )
    message = model.train(training_data)
    model.save(model_identifier)
    return message


@app.post("/train/")
def train(parameters: api_params.TrainParameters):
    """
    Endpoint that allows synchronous Kohonen grid training.

    Body Parameters
    ---------------
    parameters : api_params.TrainParameters
        See app.api_parameters.TrainParameters class for detail.

    Example
    -------
    POST @ http://mykohonenapp.com/train/

    Returns
    -------
    response : dict
        Dictionary with following key(s) :
            'message'
            'labels', 'learning_rates', 'neigh_radii', 'bmu_indicies' if verbose is true. (See Kohonen.train() for more details.)
            'operation-location' : model name/identifier under which trained model is saved.
            'exception' if exception occurs and debug_mode is true.
    """
    response = {}
    try:
        model_name = local_utils.generate_model_name(
            parameters.grid_shape, parameters.max_iterations, parameters.learning_rate
        )
        response = _train(parameters, model_identifier=model_name)
        response["operation-location"] = model_name
        status_code = HTTPCode.OK.value
    except Exception as e:
        if parameters.debug_mode:
            response["exception"] = e
        status_code = HTTPCode.INTERNAL_SERVER_ERROR.value
        _logger.exception(e, stack_info=True)
    return JSONResponse(content=response, status_code=status_code)


@app.post("/atrain/")
async def async_train(parameters: api_params.TrainParameters, background_tasks: BackgroundTasks):
    """
    Endpoint that allows asynchronous Kohonen grid training. This

    Body Parameters
    ---------------
    parameters : api_params.TrainParameters
        See app.api_parameters.TrainParameters class for detail.

    Example
    -------
    POST @ http://mykohonenapp.com/atrain/

    Returns
    -------
    response : dict
        'operation-location' : model name/identifier under which trained model is saved.
    """
    response = {}
    try:
        model_name = local_utils.generate_model_name(
            parameters.grid_shape, parameters.max_iterations, parameters.learning_rate
        )
        background_tasks.add_task(_train, parameters, model_name)
        status_code = HTTPCode.OK.value
        response["operation-location"] = model_name
    except Exception as e:
        if parameters.debug_mode:
            response["exception"] = e
        status_code = HTTPCode.INTERNAL_SERVER_ERROR.value
        _logger.exception(e, stack_info=True)
    return JSONResponse(content=response, status_code=status_code)


@app.get("/predict/")
def predict(parameters: api_params.PredictParameters):
    """
    Endpoint that allows to check associations of new data to the grid.

    Parameters: api_params.PredictParameters
        See api_params.PredictParameters class for detail.

    Example
    -------
    GET @ http://mykohonenapp.com/predict/
    """
    response = {}
    model_name = parameters.model_name
    try:
        weights = Kohonen.load(model_name)
        model = Kohonen(shape=(weights.shape[0], weights.shape[1]))
        model.set_weights(weights)
        predictions = model.predict(
            np.array(parameters.test_data).reshape(parameters.test_data_shape[0], parameters.test_data_shape[1])
        ).tolist()
        status_code = HTTPCode.OK.value
        response["operation-location"] = model_name
        response["predictions"] = predictions
    except FileNotFoundError:
        response["message"] = f"There is no model named '{model_name}' found."
        status_code = HTTPCode.NOT_FOUND.value
    except Exception as e:
        if parameters.debug_mode:
            response["exception"] = e
        status_code = HTTPCode.INTERNAL_SERVER_ERROR.value
        _logger.exception(e, stack_info=True)
    return JSONResponse(content=response, status_code=status_code)


@app.get("/list-of-models/")
def get_model_list():
    """
    Endpoint that can share all the models that are trained and saved.

    Example
    -------
    GET @ 'http://mykohonenapp.com/list-of-models/'

    Returns
    -------
    List of all file names under 'saved_models/' dir
    """
    saved_models = []
    try:
        root = MODELS_DIR
        saved_models = [f.replace(NPY_EXT, "") for f in listdir(root) if (isfile(join(root, f)) and (NPY_EXT in f))]
        status_code = HTTPCode.OK.value
    except Exception as e:
        status_code = HTTPCode.INTERNAL_SERVER_ERROR.value
        _logger.exception(e, stack_info=True)
    return JSONResponse(content=saved_models, status_code=status_code)


@app.get("/download/{model_name}/")
def download(model_name: str):
    """
    Endpoint to download model (weights)

    Parameters
    ----------
    model_name : str
        model_name (alias identifier) that uniquely identifies a trained model.

    Example
    -------
    GET @ 'http://mykohonenapp.com/download/05-03-2021_19h55m13sT_10X10_100N_0.1LR/'
    Eg. 05-03-2021_19h55m13sT_10X10_100N_0.1LR : is unique identifier / model's file name.

    All model files are saved as .npy format. However, that extension is not needed set by user.

    Returns
    -------
    Model (.npy) file under media type octet-stream.
    Example will return in a 10x10_100N.npy file downloaded on client.
    """
    response = {}
    model_path = local_utils.get_model_path(model_name)
    if not isfile(model_path):
        response["message"] = f"{model_name} could not be found."
        return JSONResponse(content=response, status_code=HTTPCode.NOT_FOUND.value)

    try:
        file = FileResponse(
            model_path,
            media_type="application/octet-stream",
            filename=f"{model_name}{NPY_EXT}",
        )
        return file
    except Exception as e:
        _logger.exception(e, stack_info=True)
        return JSONResponse(content=response, status_code=HTTPCode.INTERNAL_SERVER_ERROR.value)