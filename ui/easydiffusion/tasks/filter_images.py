import json
import pprint

from sdkit.filter import apply_filters
from sdkit.models import load_model
from sdkit.utils import img_to_base64_str, get_image, log

from easydiffusion import model_manager, runtime
from easydiffusion.types import FilterImageRequest, FilterImageResponse, ModelsData, OutputFormatData

from .task import Task


class FilterTask(Task):
    "For applying filters to input images"

    def __init__(
        self, req: FilterImageRequest, session_id: str, models_data: ModelsData, output_format: OutputFormatData
    ):
        super().__init__(session_id)

        self.request = req
        self.models_data = models_data
        self.output_format = output_format

        # convert to multi-filter format, if necessary
        if isinstance(req.filter, str):
            req.filter_params = {req.filter: req.filter_params}
            req.filter = [req.filter]

        if not isinstance(req.image, list):
            req.image = [req.image]

    def run(self):
        "Runs the image filtering task on the assigned thread"

        context = runtime.context

        model_manager.resolve_model_paths(self.models_data)
        model_manager.reload_models_if_necessary(context, self.models_data)
        model_manager.fail_if_models_did_not_load(context)

        print_task_info(self.request, self.models_data, self.output_format)

        if isinstance(self.request.image, list):
            images = [get_image(img) for img in self.request.image]
        else:
            images = get_image(self.request.image)

        images = filter_images(context, images, self.request.filter, self.request.filter_params)

        output_format = self.output_format
        images = [
            img_to_base64_str(
                img, output_format.output_format, output_format.output_quality, output_format.output_lossless
            )
            for img in images
        ]

        res = FilterImageResponse(self.request, self.models_data, images=images)
        res = res.json()
        self.buffer_queue.put(json.dumps(res))
        log.info("Filter task completed")

        self.response = res


def filter_images(context, images, filters, filter_params={}):
    filters = filters if isinstance(filters, list) else [filters]

    for filter_name in filters:
        params = filter_params.get(filter_name, {})

        previous_state = before_filter(context, filter_name, params)

        try:
            images = apply_filters(context, filter_name, images, **params)
        finally:
            after_filter(context, filter_name, params, previous_state)

    return images


def before_filter(context, filter_name, filter_params):
    if filter_name == "codeformer":
        from easydiffusion.model_manager import DEFAULT_MODELS, resolve_model_to_use

        default_realesrgan = DEFAULT_MODELS["realesrgan"][0]["file_name"]
        prev_realesrgan_path = None

        upscale_faces = filter_params.get("upscale_faces", False)
        if upscale_faces and default_realesrgan not in context.model_paths["realesrgan"]:
            prev_realesrgan_path = context.model_paths.get("realesrgan")
            context.model_paths["realesrgan"] = resolve_model_to_use(default_realesrgan, "realesrgan")
            load_model(context, "realesrgan")

        return prev_realesrgan_path


def after_filter(context, filter_name, filter_params, previous_state):
    if filter_name == "codeformer":
        prev_realesrgan_path = previous_state
        if prev_realesrgan_path:
            context.model_paths["realesrgan"] = prev_realesrgan_path
            load_model(context, "realesrgan")


def print_task_info(req: FilterImageRequest, models_data: ModelsData, output_format: OutputFormatData):
    req_str = pprint.pformat({"filter": req.filter, "filter_params": req.filter_params}).replace("[", "\[")
    models_data = pprint.pformat(models_data.dict()).replace("[", "\[")
    output_format = pprint.pformat(output_format.dict()).replace("[", "\[")

    log.info(f"request: {req_str}")
    log.info(f"models data: {models_data}")
    log.info(f"output format: {output_format}")
