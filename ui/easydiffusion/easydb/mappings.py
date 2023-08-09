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
    time_created = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return "<GalleryImage(path='%s', seed='%s', use_stable_diffusion_model='%s', clip_skip='%s', use_vae_model='%s', sampler_name='%s', width='%s', height='%s', num_inference_steps='%s', guidance_scale='%s', lora='%s', use_hypernetwork_model='%s', tiling='%s', use_face_correction='%s', use_upscale='%s', prompt='%s', negative_prompt='%s')>" % (
            self.path, self.seed, self.use_stable_diffusion_model, self.clip_skip, self.use_vae_model, self.sampler_name, self.width, self.height, self.num_inference_steps, self.guidance_scale, self.lora, self.use_hypernetwork_model, self.tiling, self.use_face_correction, self.use_upscale, self.prompt, self.negative_prompt)

from easydiffusion.easydb.database import engine
GalleryImage.metadata.create_all(engine)
