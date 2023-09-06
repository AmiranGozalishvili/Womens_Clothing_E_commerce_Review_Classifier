FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8
RUN apt-get update && apt-get install build-essential -y

RUN pip install --upgrade pip
RUN pip install -U nltk
RUN pip install opendatasets
RUN pip install contractions
RUN pip install autocorrect
RUN pip install openpyxl -U
RUN pip install textattack
RUN pip install tensorflow_text
RUN pip install ipywidgets


COPY ./app /app

WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8088"]
