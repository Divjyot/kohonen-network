from pydantic import BaseModel
from typing import Optional


class Parameters(BaseModel):
    """
    Base class defining common body parameters that
    are available to any endpoint exposed via this app.
    """

    debug_mode: Optional[bool] = False
    verbose: Optional[bool] = False


class TrainParameters(Parameters):
    """
    Class that declares body parameters for /train/, /atrain/ endpoints.
    """

    training_data: list
    training_data_shape: list
    grid_shape: list
    max_iterations: int
    learning_rate: Optional[float] = 0.1

    pp_FIND_BMU: Optional[bool] = False
    pp_INF_BMU_W: Optional[bool] = False
    pp_INF_BMU_POS: Optional[bool] = False


class PredictParameters(Parameters):
    """
    Class that declares body parameters for /predict/ endpoint.
    """

    model_name: str
    test_data: list
    test_data_shape: list
