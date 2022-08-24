from fastapi import FastAPI
from starlette.responses import FileResponse
from pydantic import BaseModel

import requests

LOCAL_SERVER_URL = 'http://localhost:5000'
PREDICT_URL = LOCAL_SERVER_URL + '/predictions'

app = FastAPI()

# defaults from https://huggingface.co/blog/stable_diffusion
class ImageRequest(BaseModel):
    prompt: str
    num_outputs: str = "1"
    num_inference_steps: str = "50"
    guidance_scale: str = "7.5"
    width: str = "512"
    height: str = "512"
    seed: str = "30000"

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
    data = {
        "input": {
            "prompt": req.prompt,
            "num_outputs": req.num_outputs,
            "num_inference_steps": req.num_inference_steps,
            "width": req.width,
            "height": req.height,
            "seed": req.seed,
            "guidance_scale": req.guidance_scale,
        }
    }

    if req.seed == "-1":
        del data['input']['seed']

    res = requests.post(PREDICT_URL, json=data)
    return res.json()

@app.get('/ding.mp3')
def read_root():
    return FileResponse('ding.mp3')
