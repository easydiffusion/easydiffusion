from fastapi import FastAPI
from starlette.responses import FileResponse
from pydantic import BaseModel

import requests

LOCAL_SERVER_URL = 'http://localhost:5000'
PREDICT_URL = LOCAL_SERVER_URL + '/predictions'

app = FastAPI()

class ImageRequest(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512

@app.get('/')
def read_root():
    return FileResponse('index.html')

@app.get('/ping')
async def ping():
    try:
        requests.get(LOCAL_SERVER_URL)
        return {'OK'}
    except:
        return {'ERROR'}

@app.post('/image')
async def image(req : ImageRequest):
    res = requests.post(PREDICT_URL, json={
        "input": {
            "prompt": req.prompt,
            "width": str(req.width),
            "height": str(req.height),
        }
    })
    return res.json()

@app.get('/ding.mp3')
def read_root():
    return FileResponse('ding.mp3')
