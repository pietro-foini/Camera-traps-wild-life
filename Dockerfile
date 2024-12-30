FROM python:3.9-slim as base_python
ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
WORKDIR $APP_HOME

FROM base_python as poetry
COPY . ./
RUN pip install -r requirements.txt
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN poetry lock --no-update
RUN poetry install

FROM base_python as runtime
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
ENV PATH="$APP_HOME/.venv/bin:$PATH"
COPY  --from=poetry $APP_HOME $APP_HOME

WORKDIR $APP_HOME/src
CMD exec gunicorn --bind :$PORT --workers 2 --threads 8 --timeout 0 main:app