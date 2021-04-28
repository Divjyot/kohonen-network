class KohonenException(Exception):
    """
    Base class for all Kohonen exceptions
    """
    def __init__(self, text):
        Exception.__init__(self, f"Kohonen: {text}")


class KohonenInputError(KohonenException):
    """
    Exceptions regarding incorrect training input are handled by this class. 
    """
    def __init__(self, text):
        super().__init__(f"Invalid input to model. {text}")


class KohonenMissingWeightsError(KohonenException):
    """
    Exceptions regarding missing weights are handled by this class. 
    """
    def __init__(self, text):
        super().__init__(f"No weights found. Check if model has not been trained. {text}")