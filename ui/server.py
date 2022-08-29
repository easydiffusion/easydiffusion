import traceback

import sys
import os

SCRIPT_DIR = os.getcwd()
print('started in ', SCRIPT_DIR)

SD_UI_DIR = os.getenv('SD_UI_PATH', None)
sys.path.append(os.path.dirname(SD_UI_DIR))

from fastapi import FastAPI, HTTPException
from starlette.responses import FileResponse
from pydantic import BaseModel

from sd_internal import Request, Response

app = FastAPI()

model_loaded = False
model_is_loading = False

# defaults from https://huggingface.co/blog/stable_diffusion
class ImageRequest(BaseModel, Request):
    pass

@app.get('/')
def read_root():
    return FileResponse(os.path.join(SD_UI_DIR, 'index.html'))

@app.get('/ping')
async def ping():
    global model_loaded, model_is_loading

    try:
        if model_loaded:
            return {'OK'}

        if model_is_loading:
            return {'ERROR'}

        model_is_loading = True

        from sd_internal import runtime
        runtime.load_model(ckpt="sd-v1-4.ckpt")

        model_loaded = True
        model_is_loading = False

        return {'OK'}
    except Exception as e:
        traceback.print_exception(e)
        return HTTPException(status_code=500, detail=str(e))

@app.post('/image')
async def image(req : ImageRequest):
    from sd_internal import runtime

    try:
        generator = runtime.txt2img if req.init_image is None else runtime.img2img
        res: Response = generator(req)

        return res.json()
    except Exception as e:
        traceback.print_exception(e)
        return HTTPException(status_code=500, detail=str(e))

@app.get('/media/ding.mp3')
def read_root():
    return FileResponse(os.path.join(SD_UI_DIR, 'media/ding.mp3'))

# start the browser ui
import webbrowser; webbrowser.open('http://localhost:9000')
