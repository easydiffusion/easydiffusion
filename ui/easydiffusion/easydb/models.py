from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, BLOB
from sqlalchemy.orm import relationship

from easydiffusion.easydb.database import BucketBase


class Bucket(BucketBase):
    __tablename__ = "bucket"

    id = Column(Integer, primary_key=True, index=True)
    path = Column(String, unique=True, index=True)

    bucketfiles = relationship("BucketFile", back_populates="bucket")


class BucketFile(BucketBase):
    __tablename__ = "bucketfile"

    filename = Column(String, index=True, primary_key=True)
    bucket_id = Column(Integer, ForeignKey("bucket.id"), primary_key=True)

    data = Column(BLOB, index=False)

    bucket = relationship("Bucket", back_populates="bucketfiles")

