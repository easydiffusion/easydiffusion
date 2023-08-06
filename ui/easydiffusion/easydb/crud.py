from sqlalchemy.orm import Session

from easydiffusion.easydb import models, schemas


def get_bucket_by_path(db: Session, path: str):
    return db.query(models.Bucket).filter(models.Bucket.path == path).first()


def create_bucket(db: Session, bucket: schemas.BucketCreate):
    db_bucket = models.Bucket(path=bucket.path)
    db.add(db_bucket)
    db.commit()
    db.refresh(db_bucket)
    return db_bucket


def create_bucketfile(db: Session, bucketfile: schemas.BucketFileCreate, bucket_id: int):
    db_bucketfile = models.BucketFile(**bucketfile.dict(), bucket_id=bucket_id)
    db.merge(db_bucketfile)
    db.commit()
    db_bucketfile = db.query(models.BucketFile).filter(models.BucketFile.bucket_id==bucket_id, models.BucketFile.filename==bucketfile.filename).first()
    return db_bucketfile

