from easydiffusion import app
from base64 import b32encode, b32decode
from fastapi import HTTPException
from fastapi.responses import FileResponse
import os
import mimetypes

# Manage an object store bucket
################################
# All file and directory names will be base32 encoded. This way, no special 
# character handling is required, and ../../.. can't be used to escape the
# jail. 



# get_object
#
def get_object(obj_path):
    path, filename = _get_path(obj_path)

    full_path = os.path.join(path, filename)

    if os.path.isfile(full_path):
        # If a file is requested, get its mime type and return the file
        extension = os.path.splitext(obj_path)[1]
        mime_type = mimetypes.guess_type(extension)[0]

        return FileResponse(full_path, media_type=mime_type)

    elif os.path.isdir(full_path):
        # Return directory listing 
        files = [ _b32_to_txt(x) for x in os.listdir(full_path) ]

        return {"files": files}
    else:
        raise HTTPException(status_code=404, detail="No such object was found.")

    return {"status": "OK"}

def post_object(obj_path, file):
    path, filename = _get_path(obj_path)

    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, filename),"wb") as f:
        f.write(file)

    return {"status": "OK", "path": path, "filename": filename }

def delete_object(obj_path):
    path, filename = _get_path(obj_path)
    full_path = os.path.join(path, filename)

    if os.path.isfile(full_path):
        os.remove(full_path)
        return {"status": "OK"}

    return HTTPException(status_code=404, detail="No such object was found.")

def _txt_to_b32(string):
    return b32encode(string.encode()).decode().rstrip("=")

def _b32_to_txt(string):
    return b32decode(_pad_string_with_equals(string).encode(), casefold=True).decode()

def _get_path(obj_path):
    path_elements = obj_path.split("/")
    dir_elements = [ _txt_to_b32(x) for x in path_elements[:-1] ]
    path = os.path.join(app.BUCKET_DIR, *dir_elements)
    filename = _txt_to_b32(path_elements[-1])

    return (path, filename)

def _pad_string_with_equals(string):
    remainder = len(string) % 8
    padding_length = (8 - remainder) % 8

    return string + "=" * padding_length

