import logging
import hashlib

log = logging.getLogger("easydiffusion")

from .save_utils import (
    save_images_to_disk,
    get_printable_request,
)

def sha256sum(filename):
    sha256 = hashlib.sha256()
    with open(filename, "rb") as f:
        while True:
            data = f.read(8192)  # Read in chunks of 8192 bytes
            if not data:
                break
            sha256.update(data)

    return sha256.hexdigest()

