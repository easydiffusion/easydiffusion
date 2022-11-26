from time import time
import re
import rich
import aiohttp
import asyncio
import os
from aiofile import async_open
from . import ModelDownloadRequest


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
                    if filename == "" and "Content-Disposition" in resp.headers:
                        m = re.match('filename="([^"]*)"', resp.headers["Content-Disposition"])
                        if m:
                            filename = m.group(1)
                    if filename == "":
                        filename = url.split("/")[-1]
                    mode = "wb" if offset == 0 else "ab"
                    afp = await async_open(os.path.join( folder, filename), mode)
                while True:
                    chunk = await resp.content.read(1024*1024)
                    if not chunk:
                        rich.print("[orange]not chunk[/orange]")
                        break
                    await afp.write(chunk)
                    total_size += len(chunk)
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


async def modelDownloadTask(req: ModelDownloadRequest):
    print(f"Download Task ID {id(req)}")
    await downloadFile(req.url, id(req), req.path)

