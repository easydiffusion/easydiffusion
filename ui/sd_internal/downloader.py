from time import time
import re
import rich
import aiohttp
import asyncio
import os
from aiofile import async_open
from . import ModelDownloadRequest
import traceback

downloadstatus = {}


async def downloadFile(url, id, folder, filename="", offset=0):
    afp = None
    total_size = 0
    start = time()
    last_msg = start
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=None, headers={"Range": f"bytes={str(offset)}-"}) as resp:
                # First chunk, so there's no output file opened yet. First determine the filename, than open the file
                if afp == None:
                    # If no filename was provided from the caller of this function, check for a filename in the HTTP
                    # response. If that one doesn't exist, take the last part from the URL.
                    if filename == "" and "Content-Disposition" in resp.headers:
                        rich.print(f'[yellow]Content-Disposition: {resp.headers["Content-Disposition"]}[/yellow]')
                        m = re.search(r'filename="([^"]*)"', resp.headers["Content-Disposition"])
                        if m:
                            filename = m.group(1)
                    if filename == "":
                        filename = url.split("/")[-1]
                    rich.print(f'[yellow]Filename {filename}[/yellow]')
                    mode = "wb" if offset == 0 else "ab"
                    afp = await async_open(os.path.join( folder, filename), mode)
                    if offset == 0:
                        downloadstatus[id]={"state": "in progress", "downloaded":0, "total": int(resp.headers["content-length"]), "filename":filename}
                while True:
                    chunk = await resp.content.read(1024*1024)
                    if not chunk:
                        rich.print("[orange]not chunk[/orange]")
                        break
                    await afp.write(chunk)
                    total_size += len(chunk)
                    downloadstatus[id]["downloaded"] = offset + total_size
                    if time() - last_msg > 2:
                        rich.print(f'[cyan]{time() - start:0.2f}s, downloaded: {total_size / (1024 * 1024):0.0f}MB of {int(resp.headers["content-length"])/1024/1024:0.0f}MB[/cyan]')
                        last_msg = time()
    except Exception as e:
        rich.print(f"[yellow]{traceback.format_exc()}[/yellow]")
        rich.print(f"[yellow]Exception: {e}[/yellow]")
    if afp != None:
        await afp.close()
    if total_size != 0 and total_size != int(resp.headers["content-length"]):
        # Resume the request
        await asyncio.sleep(2)
        await downloadFile(url, folder, filename, total_size)
    downloadstatus[id]["state"] = "completed"

def getDownloadProgress(id):
    return downloadstatus[id]

async def modelDownloadTask(req: ModelDownloadRequest):
    print(f"Download Task ID {id(req)}")
    await downloadFile(req.url, id(req), req.path)

