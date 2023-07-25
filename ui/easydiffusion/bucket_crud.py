from sqlalchemy.orm import Session

from easydiffusion import bucket_models, bucket_schemas


def get_bucket(db: Session, bucket_id: int):
    return db.query(bucket_models.Bucket).filter(bucket_models.Bucket.id == bucket_id).first()


def get_bucket_by_path(db: Session, path: str):
    return db.query(bucket_models.Bucket).filter(bucket_models.Bucket.path == path).first()


def get_buckets(db: Session, skip: int = 0, limit: int = 100):
    return db.query(bucket_models.Bucket).offset(skip).limit(limit).all()


def create_bucket(db: Session, bucket: bucket_schemas.BucketCreate):
    db_bucket = bucket_models.Bucket(path=bucket.path)
    db.add(db_bucket)
    db.commit()
    db.refresh(db_bucket)
    return db_bucket


def get_bucketfiles(db: Session, skip: int = 0, limit: int = 100):
    return db.query(bucket_models.BucketFile).offset(skip).limit(limit).all()


def create_bucketfile(db: Session, bucketfile: bucket_schemas.BucketFileCreate, bucket_id: int):
    db_bucketfile = bucket_models.BucketFile(**bucketfile.dict(), bucket_id=bucket_id)
    db.add(db_bucketfile)
    db.commit()
    db.refresh(db_bucketfile)
    return db_bucketfile

