import ldm.models.diffusion.ddim
import ldm.models.diffusion.plms
import torch

from easydiffusion import renderer


def register_buffer(self, name, attr):
    """
    credit: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_hijack.py
    """

    if type(attr) == torch.Tensor:
        if attr.device.type != renderer.context.device:
            attr = attr.to(device=renderer.context.device, dtype=torch.float32 if renderer.context.device == "mps" else None)

    setattr(self, name, attr)


ldm.models.diffusion.ddim.DDIMSampler.register_buffer = register_buffer
ldm.models.diffusion.plms.PLMSSampler.register_buffer = register_buffer
