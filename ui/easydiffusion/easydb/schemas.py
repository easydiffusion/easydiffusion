from typing import List, Union

from pydantic import BaseModel


class BucketFileBase(BaseModel):
    filename: str
    data: bytes


class BucketFileCreate(BucketFileBase):
    pass


class BucketFile(BucketFileBase):
    bucket_id: int

    class Config:
        orm_mode = True


class BucketBase(BaseModel):
    path: str


class BucketCreate(BucketBase):
    pass


class Bucket(BucketBase):
    id: int
    bucketfiles: List[BucketFile] = []

    class Config:
        orm_mode = True

