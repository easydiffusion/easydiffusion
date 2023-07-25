from typing import List

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from easydiffusion import bucket_crud, bucket_models, bucket_schemas
from easydiffusion.bucket_database import SessionLocal, engine


def init():
    from easydiffusion.server import server_api

    bucket_models.BucketBase.metadata.create_all(bind=engine)


    # Dependency
    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()


    @server_api.post("/buckets/", response_model=bucket_schemas.Bucket)
    def create_bucket(bucket: bucket_schemas.BucketCreate, db: Session = Depends(get_db)):
        db_bucket = bucket_crud.get_bucket_by_path(db, path=bucket.path)
        if db_bucket:
            raise HTTPException(status_code=400, detail="Bucket already exists")
        return bucket_crud.create_bucket(db=db, bucket=bucket)

    @server_api.get("/buckets/", response_model=List[bucket_schemas.Bucket])
    def read_bucket(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
        buckets = bucket_crud.get_buckets(db, skip=skip, limit=limit)
        return buckets


    @server_api.get("/buckets/{bucket_id}", response_model=bucket_schemas.Bucket)
    def read_bucket(bucket_id: int, db: Session = Depends(get_db)):
        db_bucket = bucket_crud.get_bucket(db, bucket_id=bucket_id)
        if db_bucket is None:
            raise HTTPException(status_code=404, detail="Bucket not found")
        return db_bucket


    @server_api.post("/buckets/{bucket_id}/items/", response_model=bucket_schemas.BucketFile)
    def create_bucketfile_in_bucket(
        bucket_id: int, bucketfile: bucket_schemas.BucketFileCreate, db: Session = Depends(get_db)
    ):
        return bucket_crud.create_bucketfile(db=db, bucketfile=bucketfile, bucket_id=bucket_id)


    @server_api.get("/bucketfiles/", response_model=List[bucket_schemas.BucketFile])
    def read_bucketfiles(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
        bucketfiles = bucket_crud.get_bucketfiles(db, skip=skip, limit=limit)
        return bucketfiles

