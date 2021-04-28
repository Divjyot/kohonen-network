FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8
LABEL maintainer="Divjyot Singh"

# Current directory is /app
COPY ./requirements/prod.requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
COPY ./app /app