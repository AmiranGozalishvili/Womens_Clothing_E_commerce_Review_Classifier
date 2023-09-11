from fastapi import FastAPI

from commands import commands
from opinionparse import opinion_parser
from predictpolarity import dependency_matching, polarity

app = FastAPI()
run_all = commands()

@app.get('/')
def get_root():
    return {'message': 'Welcome to ASBA API'}

@app.get('/ASBA')
def textanalyze(text):
    Opinion_text = opinion_parser(text)
    Polarity = polarity(text)
    Descriptors = dependency_matching(text)
    return ("Opinion Text", Opinion_text), ("Polarity", Polarity), ("Descriptor", Descriptors)
