
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8
RUN apt-get update && apt-get install build-essential -y

RUN pip install --upgrade pip
RUN pip install openpyxl -U
RUN pip install -U scikit-learn
RUN pip install keras
RUN pip install torch
RUN pip install textattack
RUN pip install tensorflow_text
RUN pip install -U tensorflow
RUN pip install Keras-Preprocessing
RUN pip install IPython
RUN python -c "import nltk; nltk.download('omw-1.4')"


COPY app /app

WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
