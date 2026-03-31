import os
import json
import pprint
import time

from numpy import base_repr

from sdkit.utils import img_to_base64_str, log, save_images, base64_str_to_img

from easydiffusion import model_manager, runtime
from easydiffusion.types import (
    FilterImageRequest,
    FilterImageResponse,
    ModelsData,
    OutputFormatData,
    SaveToDiskData,
    TaskData,
    GenerateImageRequest,
)
from easydiffusion.utils import filter_nsfw
from easydiffusion.utils.save_utils import format_folder_name

from .task import Task


class FilterTask(Task):
    "For applying filters to input images"

    def __init__(
        self,
        req: FilterImageRequest,
        task_data: TaskData,
        models_data: ModelsData,
        output_format: OutputFormatData,
        save_data: SaveToDiskData,
    ):
        super().__init__(task_data.session_id)

        task_data.request_id = self.id

        self.request = req
        self.task_data = task_data
        self.models_data = models_data
        self.output_format = output_format
        self.save_data = save_data

        # convert to multi-filter format, if necessary
        if isinstance(req.filter, str):
            if req.filter not in req.filter_params:
                req.filter_params = {req.filter: req.filter_params}

            req.filter = [req.filter]

        if not isinstance(req.image, list):
            req.image = [req.image]

    def run(self):
        "Runs the image filtering task on the assigned thread"

        from easydiffusion import app
        from easydiffusion.backend_manager import backend

        context = runtime.context

        model_manager.resolve_model_paths(self.models_data)
        model_manager.reload_models_if_necessary(context, self.models_data)
        model_manager.fail_if_models_did_not_load(context)

        print_task_info(self.request, self.models_data, self.output_format, self.save_data)

        has_nsfw_filter = "nsfw_filter" in self.request.filter

        output_format = self.output_format

        backend.set_options(
            context,
            output_format=output_format.output_format,
            output_quality=output_format.output_quality,
            output_lossless=output_format.output_lossless,
        )

        images = backend.filter_images(
            context, self.request.image, self.request.filter, self.request.filter_params, input_type="base64"
        )

        if has_nsfw_filter:
            images = filter_nsfw(images)

        if self.save_data.save_to_disk_path is not None:
            app_config = app.getConfig()
            folder_format = app_config.get("folder_format", "$id")

            dummy_req = GenerateImageRequest()
            img_id = base_repr(int(time.time() * 10000), 36)[-7:]  # Base 36 conversion, 0-9, A-Z

            save_dir_path = os.path.join(
                self.save_data.save_to_disk_path, format_folder_name(folder_format, dummy_req, self.task_data)
            )
            images_pil = [base64_str_to_img(img) for img in images]
            save_images(
                images_pil,
                save_dir_path,
                file_name=img_id,
                output_format=output_format.output_format,
                output_quality=output_format.output_quality,
                output_lossless=output_format.output_lossless,
            )

        res = FilterImageResponse(self.request, self.models_data, images=images)
        res = res.json()
        self.buffer_queue.put(json.dumps(res))

        log.info("Filter task completed")

        self.response = res


def print_task_info(
    req: FilterImageRequest, models_data: ModelsData, output_format: OutputFormatData, save_data: SaveToDiskData
):
    req_str = pprint.pformat({"filter": req.filter, "filter_params": req.filter_params}).replace("[", "\[")
    models_data = pprint.pformat(models_data.dict()).replace("[", "\[")
    output_format = pprint.pformat(output_format.dict()).replace("[", "\[")
    save_data = pprint.pformat(save_data.dict()).replace("[", "\[")

    log.info(f"request: {req_str}")
    log.info(f"models data: {models_data}")
    log.info(f"output format: {output_format}")
    log.info(f"save data: {save_data}")
