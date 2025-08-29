from typing import List

from fastapi import Depends, FastAPI, HTTPException, Response, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from easydiffusion.easydb import crud, models, schemas
from easydiffusion.easydb.database import SessionLocal, engine

from requests.compat import urlparse
from os.path import abspath

import base64, json

MIME_TYPES = {
    "jpg":  "image/jpeg",
    "jpeg": "image/jpeg",
    "gif":  "image/gif",
    "png":  "image/png",
    "webp": "image/webp",
    "js":   "text/javascript",
    "htm":  "text/html",
    "html": "text/html",
    "css":  "text/css",
    "json": "application/json",
    "mjs":  "application/json",
    "yaml": "application/yaml",
    "svg":  "image/svg+xml",
    "txt":  "text/plain",
}

def init():
    from easydiffusion.server import server_api

    models.BucketBase.metadata.create_all(bind=engine)


    # Dependency
    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    @server_api.get("/bucket/{obj_path:path}")
    def bucket_get_object(obj_path: str, db: Session = Depends(get_db)):
        filename = get_filename_from_url(obj_path)
        path = get_path_from_url(obj_path)

        if filename==None:
            bucket = crud.get_bucket_by_path(db, path=path)
            if bucket == None:
                raise HTTPException(status_code=404, detail="Bucket not found")
            bucketfiles = db.query(models.BucketFile).with_entities(models.BucketFile.filename).filter(models.BucketFile.bucket_id == bucket.id).all()
            bucketfiles = [ x.filename for x in bucketfiles ]
            return bucketfiles

        else:
            bucket = crud.get_bucket_by_path(db, path)
            if bucket == None:
                raise HTTPException(status_code=404, detail="Bucket not found")
            bucket_id = bucket.id
            bucketfile = db.query(models.BucketFile).filter(models.BucketFile.bucket_id == bucket_id, models.BucketFile.filename == filename).first()
            if bucketfile == None:
                raise HTTPException(status_code=404, detail="File not found")

            suffix = get_suffix_from_filename(filename)

            return Response(content=bucketfile.data, media_type=MIME_TYPES.get(suffix, "application/octet-stream")) 

    @server_api.post("/bucket/{obj_path:path}")
    def bucket_post_object(obj_path: str, file: bytes = File(), db: Session = Depends(get_db)):
        filename = get_filename_from_url(obj_path)
        path = get_path_from_url(obj_path)
        bucket = crud.get_bucket_by_path(db, path)

        if bucket == None:
            bucket = crud.create_bucket(db=db, bucket=schemas.BucketCreate(path=path))
        bucket_id = bucket.id

        bucketfile = schemas.BucketFileCreate(filename=filename, data=file)
        result = crud.create_bucketfile(db=db, bucketfile=bucketfile, bucket_id=bucket_id)
        result.data = base64.encodestring(result.data)
        return result


    @server_api.post("/buckets/{bucket_id}/items/", response_model=schemas.BucketFile)
    def create_bucketfile_in_bucket(
        bucket_id: int, bucketfile: schemas.BucketFileCreate, db: Session = Depends(get_db)
    ):
        bucketfile.data = base64.decodestring(bucketfile.data)
        result =  crud.create_bucketfile(db=db, bucketfile=bucketfile, bucket_id=bucket_id)
        result.data = base64.encodestring(result.data)
        return result

    @server_api.get("/image/{image_path:path}")
    def get_image(image_path: str, db: Session = Depends(get_db)):
        from easydiffusion.easydb.mappings import GalleryImage
        image_path = str(abspath(image_path))
        try:
            image = db.query(GalleryImage).filter(GalleryImage.path == image_path).first()
            return FileResponse(image.path)
        except Exception as e:
            raise HTTPException(status_code=404, detail="Image not found")
    
    @server_api.get("/all_images")
    def get_all_images(prompt: str = "", model: str = "", page: int = 0, images_per_page: int = 50, workspace : str = "default", db: Session = Depends(get_db)):
        from easydiffusion.easydb.mappings import GalleryImage
        images = db.query(GalleryImage).filter(GalleryImage.workspace == workspace).order_by(GalleryImage.time_created.desc())
        if prompt != "":
            images = images.filter(GalleryImage.prompt.like("%"+prompt+"%"))
        if model != "":
            images = images.filter(GalleryImage.use_stable_diffusion_model.like("%"+model+"%"))
        images = images.offset(page*images_per_page).limit(images_per_page)
        return images.all()

    
def get_filename_from_url(url):
    path = urlparse(url).path
    name = path[path.rfind('/')+1:]
    return name or None 

def get_path_from_url(url):
    path = urlparse(url).path
    path = path[0:path.rfind('/')]
    return path or None 

def get_suffix_from_filename(filename):
    return filename[filename.rfind('.')+1:]
