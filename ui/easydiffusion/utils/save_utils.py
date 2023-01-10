import os
import time
import base64
import re

from easydiffusion.types import TaskData, GenerateImageRequest

from sdkit.utils import save_images, save_dicts

filename_regex = re.compile('[^a-zA-Z0-9]')

# keep in sync with `ui/media/js/dnd.js`
TASK_TEXT_MAPPING = {
    'prompt': 'Prompt',
    'width': 'Width',
    'height': 'Height',
    'seed': 'Seed',
    'num_inference_steps': 'Steps',
    'guidance_scale': 'Guidance Scale',
    'prompt_strength': 'Prompt Strength',
    'use_face_correction': 'Use Face Correction',
    'use_upscale': 'Use Upscaling',
    'upscale_amount': 'Upscale By',
    'sampler_name': 'Sampler',
    'negative_prompt': 'Negative Prompt',
    'use_stable_diffusion_model': 'Stable Diffusion model',
    'use_hypernetwork_model': 'Hypernetwork model',
    'hypernetwork_strength': 'Hypernetwork Strength'
}

def save_images_to_disk(images: list, filtered_images: list, req: GenerateImageRequest, task_data: TaskData):
    now = time.time()
    save_dir_path = os.path.join(task_data.save_to_disk_path, filename_regex.sub('_', task_data.session_id))
    metadata_entries = get_metadata_entries_for_request(req, task_data)
    make_filename = make_filename_callback(req, now=now)

    if task_data.show_only_filtered_image or filtered_images is images:
        save_images(filtered_images, save_dir_path, file_name=make_filename, output_format=task_data.output_format, output_quality=task_data.output_quality)
        save_dicts(metadata_entries, save_dir_path, file_name=make_filename, output_format=task_data.metadata_output_format)
    else:
        make_filter_filename = make_filename_callback(req, now=now, suffix='filtered')

        save_images(images, save_dir_path, file_name=make_filename, output_format=task_data.output_format, output_quality=task_data.output_quality)
        save_images(filtered_images, save_dir_path, file_name=make_filter_filename, output_format=task_data.output_format, output_quality=task_data.output_quality)
        save_dicts(metadata_entries, save_dir_path, file_name=make_filter_filename, output_format=task_data.metadata_output_format)

def get_metadata_entries_for_request(req: GenerateImageRequest, task_data: TaskData):
    metadata = get_printable_request(req)
    metadata.update({
        'use_stable_diffusion_model': task_data.use_stable_diffusion_model,
        'use_vae_model': task_data.use_vae_model,
        'use_hypernetwork_model': task_data.use_hypernetwork_model,
        'use_face_correction': task_data.use_face_correction,
        'use_upscale': task_data.use_upscale,
    })
    if metadata['use_upscale'] is not None:
        metadata['upscale_amount'] = task_data.upscale_amount

    # if text, format it in the text format expected by the UI
    is_txt_format = (task_data.metadata_output_format.lower() == 'txt')
    if is_txt_format:
        metadata = {TASK_TEXT_MAPPING[key]: val for key, val in metadata.items() if key in TASK_TEXT_MAPPING}

    entries = [metadata.copy() for _ in range(req.num_outputs)]
    for i, entry in enumerate(entries):
        entry['Seed' if is_txt_format else 'seed'] = req.seed + i

    return entries

def get_printable_request(req: GenerateImageRequest):
    metadata = req.dict()
    del metadata['init_image']
    del metadata['init_image_mask']
    return metadata

def make_filename_callback(req: GenerateImageRequest, suffix=None, now=None):
    if now is None:
        now = time.time()
    def make_filename(i):
        img_id = base64.b64encode(int(now+i).to_bytes(8, 'big')).decode() # Generate unique ID based on time.
        img_id = img_id.translate({43:None, 47:None, 61:None})[-8:] # Remove + / = and keep last 8 chars.

        prompt_flattened = filename_regex.sub('_', req.prompt)[:50]
        name = f"{prompt_flattened}_{img_id}"
        name = name if suffix is None else f'{name}_{suffix}'
        return name

    return make_filename
