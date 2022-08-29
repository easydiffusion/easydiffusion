from fastapi import FastAPI, HTTPException
from starlette.responses import FileResponse
from pydantic import BaseModel

import requests

app = FastAPI()

# defaults from https://huggingface.co/blog/stable_diffusion
class ImageRequest(BaseModel):
    prompt: str
    init_image: str = None # base64
    mask: str = None # base64
    num_outputs: str = "1"
    num_inference_steps: str = "50"
    guidance_scale: str = "7.5"
    width: str = "512"
    height: str = "512"
    seed: str = "30000"
    prompt_strength: str = "0.8"

@app.get('/')
def read_root():
    return FileResponse('index.html')

@app.get('/ping')
async def ping():
    try:
        # check if SD is present
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

    if req.init_image is not None:
        data['input']['init_image'] = req.init_image
        data['input']['prompt_strength'] = req.prompt_strength

        if req.mask is not None:
            data['input']['mask'] = req.mask

    if req.seed == "-1":
        del data['input']['seed']

    return {'OK'}

    # res = requests.post(PREDICT_URL, json=data)
    # if res.status_code != 200:
    #     raise HTTPException(status_code=500, detail=res.text)

    # return res.json()

@app.get('/media/ding.mp3')
def read_root():
    return FileResponse('media/ding.mp3')

# start the browser ui
import webbrowser; webbrowser.open('http://localhost:9000')
