# possibly move this to sdkit in the future
import os

# mirror of https://huggingface.co/AdamCodd/vit-base-nsfw-detector/blob/main/onnx/model_quantized.onnx
NSFW_MODEL_URL = (
    "https://github.com/easydiffusion/sdkit-test-data/releases/download/assets/vit-base-nsfw-detector-quantized.onnx"
)
MODEL_HASH_QUICK = "220123559305b1b07b7a0894c3471e34dccd090d71cdf337dd8012f9e40d6c28"

nsfw_check_model = None


def filter_nsfw(images, blur_radius: float = 75, print_log=True):
    global nsfw_check_model

    from easydiffusion.app import MODELS_DIR
    from sdkit.utils import base64_str_to_img, img_to_base64_str, download_file, log, hash_file_quick

    import onnxruntime as ort
    from PIL import ImageFilter
    import numpy as np

    if nsfw_check_model is None:
        model_dir = os.path.join(MODELS_DIR, "nsfw-checker")
        model_path = os.path.join(model_dir, "vit-base-nsfw-detector-quantized.onnx")

        os.makedirs(model_dir, exist_ok=True)

        if not os.path.exists(model_path) or hash_file_quick(model_path) != MODEL_HASH_QUICK:
            download_file(NSFW_MODEL_URL, model_path)

        nsfw_check_model = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Preprocess the input image
    def preprocess_image(img):
        img = img.convert("RGB")

        # config based on based on https://huggingface.co/AdamCodd/vit-base-nsfw-detector/blob/main/onnx/preprocessor_config.json
        # Resize the image
        img = img.resize((384, 384))

        # Normalize the image
        img = np.array(img) / 255.0  # Scale pixel values to [0, 1]
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        img = (img - mean) / std

        # Transpose to match input shape (batch_size, channels, height, width)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img

    # Run inference
    input_name = nsfw_check_model.get_inputs()[0].name
    output_name = nsfw_check_model.get_outputs()[0].name

    if print_log:
        log.info("Running NSFW checker (onnx)")

    results = []
    for img in images:
        is_base64 = isinstance(img, str)

        input_img = base64_str_to_img(img) if is_base64 else img

        result = nsfw_check_model.run([output_name], {input_name: preprocess_image(input_img)})
        is_nsfw = [np.argmax(arr) == 1 for arr in result][0]

        if is_nsfw:
            output_img = input_img.filter(ImageFilter.GaussianBlur(blur_radius))
            output_img = img_to_base64_str(output_img) if is_base64 else output_img
        else:
            output_img = img

        results.append(output_img)

    return results
