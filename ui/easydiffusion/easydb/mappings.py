from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class GalleryImage(Base):
    __tablename__ = 'images'

    path = Column(String, primary_key=True)
    seed = Column(Integer)
    use_stable_diffusion_model = Column(String)
    clip_skip = Column(Boolean)
    use_vae_model = Column(String)
    sampler_name = Column(String)
    width = Column(Integer)
    height = Column(Integer)
    num_inference_steps = Column(Integer)
    guidance_scale = Column(Float)
    lora = Column(String)
    use_hypernetwork_model = Column(String)
    tiling = Column(String)
    use_face_correction = Column(String)
    use_upscale = Column(String)
    prompt = Column(String)
    negative_prompt = Column(String)
    workspace = Column(String, server_default="default")
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    nsfw = Column(Boolean, server_default=None)
    scheduler_name = Column(String, server_default="")
    use_text_encoder_model = Column(String, server_default="")

from easydiffusion.easydb.database import engine
GalleryImage.metadata.create_all(engine)
