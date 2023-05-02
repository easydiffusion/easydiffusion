// placeholder until a more formal and legal-sounding privacy policy document is written. but the information below is true.

This is a summary of whether Easy Diffusion uses your data or tracks you:
* The short answer is - Easy Diffusion does *not* use your data, and does *not* track you.
* Easy Diffusion does not send your prompts or usage or analytics to anyone. There is no tracking. We don't even know how many people use Easy Diffusion, let alone their prompts.
* Easy Diffusion fetches updates to the code whenever it starts up. It does this by contacting GitHub directly, via SSL (secure connection). Only your computer and GitHub and [this repository](https://github.com/cmdr2/stable-diffusion-ui) are involved, and no third party is involved. Some countries intercepts SSL connections, that's not something we can do much about. GitHub does *not* share statistics (even with me) about how many people fetched code updates.
* Easy Diffusion fetches the models from huggingface.co and github.com, if they don't exist on your PC. For e.g. if the safety checker (NSFW) model doesn't exist, it'll try to download it.
* Easy Diffusion fetches code packages from pypi.org, which is the standard hosting service for all Python projects. That's where packages installed via `pip install` are stored.
* Occasionally, antivirus software are known to *incorrectly* flag and delete some model files, which will result in Easy Diffusion re-downloading `pytorch_model.bin`. This *incorrect deletion* affects other Stable Diffusion UIs as well, like Invoke AI - https://itch.io/post/7509488
