import os
from easydiffusion import app

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

os.makedirs(app.BUCKET_DIR, exist_ok=True)
SQLALCHEMY_DATABASE_URL = "sqlite:///"+os.path.join(app.BUCKET_DIR, "bucket.db")

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

BucketBase = declarative_base()
