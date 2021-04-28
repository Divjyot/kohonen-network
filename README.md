## Kohonen Challenge

Please see [kohonen.ipynb](kohonen.ipynb)

## Challenge Accepted!
#### For more discussion, please see [kohonen.ipynb](kohonen.ipynb)

Running from source code:
------------------------

#### As local server:
1. ```cd app/```
2. ```uvicorn main:app``` (default port is 8000)

#### As local via python file:
- Look for ```app/test.py``` <a href='app/test.py'>test.py</a>  file
- It has sample tests written for running the network over following configurations:
    1. 10X10 for 100N
    2. 10X10 for 200N
    3. 10X10 for 500N
    4. 100X100 for 1000N


Build/Running containised (on 8000/your choice port):
-----------------------------------------------------
1. ```docker build . -t kohonen``` (assuming your current directory is this project's directory)
2. ```docker run -d --name kohonen -p 8000:80  kohonen:latest```

Project Structure
------------------

<div style='color:#FF4500;'>

- ```numpy``` is used as primary library to hold and manipluate data.
- ```fastAPI``` is used to package the application and expose ```/train/```, ```/atrain/```, ```/list-of-models/```, ```/download/```, ```/predict/``` endpoints. ```fastAPI``` was chosen due to its high performance and auto tuned for number of CPU cores for handling high request load.
- The solution is packaged as a production ready server application & containerised. The solution files reside under ```app/``` folder:    
    - `kohonen.py` : <i>file contains ```Kohonen``` that implements the algorithm.</i>
    - `main.py` : <i>file act as entry point for ```fastAPI```, where all REST API requests will fall.</i>
    - `settings.py` : <i>file contain settings, contant values etc used in the project.</i>
    - `utils/utils.py` : <i>file contains helper methods that are used in the project.</i>
    - `saved_models/` : <i>folder which (can) contain saved models (weights) post training. Ideally, models/weights could be saved on a blob storage for better scalablity.</i>
        - <i>There are already saved grids/model files (.npy) under following names/configurations:</i> 
            - ```10X10 100N```
            - ```10X10 200N```
            - ```10X10 500N```
            - ```100X100 1000N```
    - `exceptions.py` : <i>file implementing kohonen algorithm exceptions.</i>
    - `logs/logging.log` : <i>file contains logs that are captured throughout the appplication.</i>
    - `api_params` : <i> file contains multiple classes that are responsible for parsing ```fastAPI``` requests (body parameters).</i>
    - `app/test.py` : <i> Please look for <a href='app/test.py'>test.py</a> file for triggering tests. Uncomment lines, to run via command line ```python test.py```</i>
    - `saved_plots/` : <i>folder to save any plot images (used in test.py) for analysis/code-testing purposes</i>
    - `saved_train_inputs/` : <i>folder to save any training data (used in test.py) as .npy file for analysis/code-testing purposes</i>
    - `configs/`  : <i>folder contains python package requiements files.</i>
        - ```prod.requirements.text``` is used to install requirements when packaging, containerising.
        - ```requirements.text``` has to be used to create local envrionment for running, testing the application via command line, jupyter lab etc.
            - (Core Libs)
                - ```! pip install  numpy==1.20.1```
                - ```! pip install  matplotlib==3.3.4```
            -  (Packaging/Productioning Libs)
                - ```! pip install fastapi==0.63.0```
                - ```! pip install aiofiles==0.6.0```
                - ```! pip install uvicorn==0.13.4```
            -   (For making HTTP requests to running application)
                - ```! pip install requests==2.25.1```
    - `Dockerfile` : <i>to package this project as REST API</i>
        - This is based on ```tiangolo/uvicorn-gunicorn-fastapi:latest``` image which has ```python==3.8.6``` preinstalled.
</div>