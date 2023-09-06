from fastapi import FastAPI

from text_augmentation import recommender_model
from commands import commands

app = FastAPI()

run_all = commands()

@app.get('/')
def get_root():
	return {'message': 'Welcome to Text Augmentation APi'}


@app.get('/Generate text for Balanced dataset')
def texttotest(text):

	return recommender_model(text)

# text example from dataset ('these pants are beautiful but very sheer and very delicate. not ideal for a mom with kids')

