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
    nsfw = Column(String, server_default='unknown')

    def __repr__(self):
        return "<GalleryImage(path='%s', seed='%s', use_stable_diffusion_model='%s', clip_skip='%s', use_vae_model='%s', sampler_name='%s', width='%s', height='%s', num_inference_steps='%s', guidance_scale='%s', lora='%s', use_hypernetwork_model='%s', tiling='%s', use_face_correction='%s', use_upscale='%s', prompt='%s', negative_prompt='%s')>" % (
            self.path, self.seed, self.use_stable_diffusion_model, self.clip_skip, self.use_vae_model, self.sampler_name, self.width, self.height, self.num_inference_steps, self.guidance_scale, self.lora, self.use_hypernetwork_model, self.tiling, self.use_face_correction, self.use_upscale, self.prompt, self.negative_prompt)

    def htmlForm(self) -> str:
        return "<div><p>Path: " + str(self.path) + "</p>" + \
                "Seed: " + str(self.seed) + "</p>" + \
                "Stable Diffusion Model: " + str(self.use_stable_diffusion_model) + "</p>" + \
                "Prompt: " + str(self.prompt) + "</p>" + \
                "Negative Prompt: " + str(self.negative_prompt) + "</p>" + \
                "Clip Skip: " + str(self.clip_skip) + "</p>" + \
                "VAE Model: " + str(self.use_vae_model) + "</p>" + \
                "Sampler: " + str(self.sampler_name) + "</p>" + \
                "Size: " + str(self.height) + "x" + str(self.width) + "</p>" + \
                "Inference Steps: " + str(self.num_inference_steps) + "</p>" + \
                "Guidance Scale: " + str(self.guidance_scale) + "</p>" + \
                "LoRA: " + str(self.lora) + "</p>" + \
                "Hypernetwork: " + str(self.use_hypernetwork_model) + "</p>" + \
                "Tiling: " + str(self.tiling) + "</p>" + \
                "Face Correction: " + str(self.use_face_correction) + "</p>" + \
                "Upscale: " + str(self.use_upscale) + "</p>" + \
                "Time Created: " + str(self.time_created) + "</p>" + \
                "NSFW: " + str(self.nsfw) + "</p></div>"


from easydiffusion.easydb.database import engine
GalleryImage.metadata.create_all(engine)
