import json
import struct
import sys

# based on the key name and logic from diffusers
# source: https://github.com/huggingface/diffusers/blob/03c3f69aa57a6cc2c995d41ea484195d719a240a/src/diffusers/loaders/single_file_utils.py#L62
# $ for f in $(find . | grep -E 'safetensors|sft|gguf'); do echo "$f - "$(python /path/to/model_identifier.py "$f"); done

"""
FIXME - These are wrong:
./2.1/stable_diffusion-ema-pruned-v2-1_768.q4_1.gguf - sd_v1_base
./2.0/512-depth-ema.safetensors - sd_v2_base
./2.1/v2-1_512-ema-pruned.safetensors - sd_v2_base
./2.1/v2-1_768-ema-pruned.safetensors - sd_v2_base
./2.0/768-v-ema.safetensors - sd_v2_base
./3.0/sd3_medium.safetensors - sd_v3_base
./3.0/sd3_medium_incl_clips.safetensors - sd_v3_base
./3.0/sd3_medium_incl_clips_t5xxlfp8.safetensors - sd_v3_base
"""

CHECKPOINT_KEY_NAMES = {
    "v1": "model.diffusion_model.output_blocks.11.0.skip_connection.weight",
    "v2": "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight",
    "xl_base": "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias",
    "xl_refiner": "conditioner.embedders.0.model.transformer.resblocks.9.mlp.c_proj.bias",
    "upscale": "model.diffusion_model.input_blocks.10.0.skip_connection.bias",
    "controlnet": [
        "control_model.time_embed.0.weight",
        "controlnet_cond_embedding.conv_in.weight",
    ],
    "controlnet_xl": "add_embedding.linear_1.weight",
    "controlnet_xl_large": "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.weight",
    "controlnet_xl_mid": "down_blocks.1.attentions.0.norm.weight",
    "playground-v2-5": "edm_mean",
    "inpainting": "model.diffusion_model.input_blocks.0.0.weight",
    "clip": "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight",
    "clip_sdxl": "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight",
    "clip_sd3": "text_encoders.clip_l.transformer.text_model.embeddings.position_embedding.weight",
    "open_clip": "cond_stage_model.model.token_embedding.weight",
    "open_clip_sdxl": "conditioner.embedders.1.model.positional_embedding",
    "open_clip_sdxl_refiner": "conditioner.embedders.0.model.text_projection",
    "open_clip_sd3": "text_encoders.clip_g.transformer.text_model.embeddings.position_embedding.weight",
    "stable_cascade_stage_b": "down_blocks.1.0.channelwise.0.weight",
    "stable_cascade_stage_c": "clip_txt_mapper.weight",
    "sd3": [
        "joint_blocks.0.context_block.adaLN_modulation.1.bias",
        "model.diffusion_model.joint_blocks.0.context_block.adaLN_modulation.1.bias",
    ],
    "sd35_large": [
        "joint_blocks.37.x_block.mlp.fc1.weight",
        "model.diffusion_model.joint_blocks.37.x_block.mlp.fc1.weight",
    ],
    "animatediff": "down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.0.pos_encoder.pe",
    "animatediff_v2": "mid_block.motion_modules.0.temporal_transformer.norm.bias",
    "animatediff_sdxl_beta": "up_blocks.2.motion_modules.0.temporal_transformer.norm.weight",
    "animatediff_scribble": "controlnet_cond_embedding.conv_in.weight",
    "animatediff_rgb": "controlnet_cond_embedding.weight",
    "auraflow": [
        "double_layers.0.attn.w2q.weight",
        "double_layers.0.attn.w1q.weight",
        "cond_seq_linear.weight",
        "t_embedder.mlp.0.weight",
    ],
    "flux": [
        "double_blocks.0.img_attn.norm.key_norm.scale",
        "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale",
    ],
    "ltx-video": [
        "model.diffusion_model.patchify_proj.weight",
        "model.diffusion_model.transformer_blocks.27.scale_shift_table",
        "patchify_proj.weight",
        "transformer_blocks.27.scale_shift_table",
        "vae.per_channel_statistics.mean-of-means",
    ],
    "autoencoder-dc": "decoder.stages.1.op_list.0.main.conv.conv.bias",
    "autoencoder-dc-sana": "encoder.project_in.conv.bias",
    "mochi-1-preview": ["model.diffusion_model.blocks.0.attn.qkv_x.weight", "blocks.0.attn.qkv_x.weight"],
    "hunyuan-video": "txt_in.individual_token_refiner.blocks.0.adaLN_modulation.1.bias",
    "instruct-pix2pix": "model.diffusion_model.input_blocks.0.0.weight",
    "lumina2": ["model.diffusion_model.cap_embedder.0.weight", "cap_embedder.0.weight"],
    "sana": [
        "blocks.0.cross_attn.q_linear.weight",
        "blocks.0.cross_attn.q_linear.bias",
        "blocks.0.cross_attn.kv_linear.weight",
        "blocks.0.cross_attn.kv_linear.bias",
    ],
    "wan": ["model.diffusion_model.head.modulation", "head.modulation"],
    "wan_vae": "decoder.middle.0.residual.0.gamma",
    "wan_vace": "vace_blocks.0.after_proj.bias",
    "hidream": "double_stream_blocks.0.block.adaLN_modulation.1.bias",
    "cosmos-1.0": [
        "net.x_embedder.proj.1.weight",
        "net.blocks.block1.blocks.0.block.attn.to_q.0.weight",
        "net.extra_pos_embedder.pos_emb_h",
    ],
    "cosmos-2.0": [
        "net.x_embedder.proj.1.weight",
        "net.blocks.0.self_attn.q_proj.weight",
        "net.pos_embedder.dim_spatial_range",
    ],
}


def read_safetensors_header(path):
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_data = f.read(header_size)
    return json.loads(header_data)


def read_gguf_header(path):
    header = {}
    with open(path, "rb") as f:
        # 1. Read GGUF Header
        magic = f.read(4)  # Read 4 bytes as the magic number
        if magic != b"GGUF":  # Compare with the byte string 'GGUF'
            raise ValueError("Invalid GGUF file: Incorrect magic number.")

        # Version (uint32_t)
        version = struct.unpack("<I", f.read(4))[0]

        # Tensor Count (uint64_t)
        n_tensors = struct.unpack("<Q", f.read(8))[0]

        # Metadata KV Count (uint64_t)
        n_kv = struct.unpack("<Q", f.read(8))[0]

        # 2. Skip Metadata (Key-Value Store) - We're only interested in tensor shapes
        for _ in range(n_kv):
            # Read key string length (uint64_t)
            key_len = struct.unpack("<Q", f.read(8))[0]
            # Read key string (char* data)
            f.read(key_len)

            # Read value type (enum gguf_type)
            value_type = struct.unpack("<I", f.read(4))[0]

            # Based on value_type, skip appropriate number of bytes.
            # Only common types are listed here. Expand for other types if needed.
            if value_type in [0, 1, 7]:  # GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_BOOL
                f.read(1)
            elif value_type in [2, 3]:  # GGUF_TYPE_UINT16, GGUF_TYPE_INT16
                f.read(2)
            elif value_type in [4, 5, 6]:  # GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32
                f.read(4)
            elif value_type in [10, 11, 12]:  # GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64
                f.read(8)
            elif value_type == 8:  # GGUF_TYPE_STRING
                str_len = struct.unpack("<Q", f.read(8))[0]
                f.read(str_len)
            elif value_type == 9:  # GGUF_TYPE_ARRAY
                array_type = struct.unpack("<I", f.read(4))[0]
                array_len = struct.unpack("<Q", f.read(8))[0]
                # Skip array data based on array_type and array_len
                # Handle nested arrays or more complex types if needed
                if array_type in [0, 1, 7]:
                    f.read(array_len * 1)
                elif array_type in [2, 3]:
                    f.read(array_len * 2)
                elif array_type in [4, 5, 6]:
                    f.read(array_len * 4)
                elif array_type in [10, 11, 12]:
                    f.read(array_len * 8)
                elif array_type == 8:  # Handle string arrays
                    for _ in range(array_len):
                        str_len = struct.unpack("<Q", f.read(8))[0]
                        f.read(str_len)
            else:
                # Handle unknown types or raise an error if strictness is needed
                pass

        # 3. Read Tensor Information
        for _ in range(n_tensors):
            # Tensor Name (gguf_str)
            name_len = struct.unpack("<Q", f.read(8))[0]
            name = f.read(name_len).decode("utf-8")

            # Number of Dimensions (uint32_t)
            n_dims = struct.unpack("<I", f.read(4))[0]

            # Shape (uint64_t[GGUF_MAX_DIMS])
            dims = []
            for _ in range(n_dims):
                dims.append(struct.unpack("<Q", f.read(8))[0])

            # Type (enum ggml_type)
            tensor_type = struct.unpack("<I", f.read(4))[0]

            # Offset (uint64_t) -  Offset from start of `data`
            offset = struct.unpack("<Q", f.read(8))[0]

            header[name] = {"shape": tuple(dims)}

    return header


def shape_of(header, key):
    """Return shape tuple for a tensor key from header or None."""
    if key not in header:
        return None
    return tuple(header[key]["shape"])


def has_any_key(header, keys):
    return any(k in header for k in (keys if isinstance(keys, list) else [keys]))


def has_all_keys(header, keys):
    return all(k in header for k in (keys if isinstance(keys, list) else [keys]))


def infer_diffusers_model_type(header):
    s = shape_of

    if CHECKPOINT_KEY_NAMES["inpainting"] in header and s(header, CHECKPOINT_KEY_NAMES["inpainting"])[1] == 9:
        if CHECKPOINT_KEY_NAMES["v2"] in header and s(header, CHECKPOINT_KEY_NAMES["v2"])[-1] == 1024:
            return "sd_v2_inpainting"
        elif CHECKPOINT_KEY_NAMES["xl_base"] in header:
            return "sd_xl_inpainting"
        else:
            return "sd_v1_inpainting"

    elif CHECKPOINT_KEY_NAMES["v2"] in header and s(header, CHECKPOINT_KEY_NAMES["v2"])[-1] == 1024:
        return "sd_v2_base"

    elif CHECKPOINT_KEY_NAMES["playground-v2-5"] in header:
        return "playground_v2_5"

    elif CHECKPOINT_KEY_NAMES["xl_base"] in header:
        return "sd_xl_base"

    elif CHECKPOINT_KEY_NAMES["xl_refiner"] in header:
        return "sd_xl_refiner"

    elif CHECKPOINT_KEY_NAMES["upscale"] in header:
        return "sd_v2_upscale"

    elif has_any_key(header, CHECKPOINT_KEY_NAMES["controlnet"]):
        if CHECKPOINT_KEY_NAMES["controlnet_xl"] in header:
            if CHECKPOINT_KEY_NAMES["controlnet_xl_large"] in header:
                return "controlnet_xl_large"
            elif CHECKPOINT_KEY_NAMES["controlnet_xl_mid"] in header:
                return "controlnet_xl_mid"
            else:
                return "controlnet_xl_small"
        else:
            return "controlnet"

    elif (
        CHECKPOINT_KEY_NAMES["stable_cascade_stage_c"] in header
        and s(header, CHECKPOINT_KEY_NAMES["stable_cascade_stage_c"])[0] == 1536
    ):
        return "stable_cascade_c_lite"
    elif (
        CHECKPOINT_KEY_NAMES["stable_cascade_stage_c"] in header
        and s(header, CHECKPOINT_KEY_NAMES["stable_cascade_stage_c"])[0] == 2048
    ):
        return "stable_cascade_c"
    elif (
        CHECKPOINT_KEY_NAMES["stable_cascade_stage_b"] in header
        and s(header, CHECKPOINT_KEY_NAMES["stable_cascade_stage_b"])[-1] == 576
    ):
        return "stable_cascade_b_lite"
    elif (
        CHECKPOINT_KEY_NAMES["stable_cascade_stage_b"] in header
        and s(header, CHECKPOINT_KEY_NAMES["stable_cascade_stage_b"])[-1] == 640
    ):
        return "stable_cascade_b"

    elif has_any_key(header, CHECKPOINT_KEY_NAMES["sd3"]) and any(
        s(header, k)[-1] == 9216 if k in header else False for k in CHECKPOINT_KEY_NAMES["sd3"]
    ):
        key = "model.diffusion_model.pos_embed" if "model.diffusion_model.pos_embed" in header else "pos_embed"
        if s(header, key)[1] == 36864:
            return "sd_v3_base"
        elif s(header, key)[1] == 147456:
            return "sd_v3_5_medium"

    elif has_any_key(header, CHECKPOINT_KEY_NAMES["sd35_large"]):
        return "sd_v3_5_large"

    elif CHECKPOINT_KEY_NAMES["animatediff"] in header:
        if CHECKPOINT_KEY_NAMES["animatediff_scribble"] in header:
            return "animatediff_scribble"
        elif CHECKPOINT_KEY_NAMES["animatediff_rgb"] in header:
            return "animatediff_rgb"
        elif CHECKPOINT_KEY_NAMES["animatediff_v2"] in header:
            return "animatediff_v2"
        elif s(header, CHECKPOINT_KEY_NAMES["animatediff_sdxl_beta"])[-1] == 320:
            return "animatediff_sdxl_beta"
        elif s(header, CHECKPOINT_KEY_NAMES["animatediff"])[1] == 24:
            return "animatediff_v1"
        else:
            return "animatediff_v3"

    elif has_any_key(header, CHECKPOINT_KEY_NAMES["flux"]):
        if "distilled_guidance_layer.layers.0.in_layer.bias" in header:
            return "chroma"

        g_in = "model.diffusion_model.guidance_in.in_layer.bias"
        if g_in in header or "guidance_in.in_layer.bias" in header:
            k = (
                "model.diffusion_model.img_in.weight"
                if "model.diffusion_model.img_in.weight" in header
                else "img_in.weight"
            )
            if s(header, k)[1] == 384:
                return "flux_fill"
            elif s(header, k)[1] == 128:
                return "flux_depth"
            else:
                return "flux_dev"
        else:
            return "flux_schnell"

    elif has_any_key(header, CHECKPOINT_KEY_NAMES["ltx-video"]):
        has_vae = "vae.encoder.conv_in.conv.bias" in header
        if any(k.endswith("transformer_blocks.47.scale_shift_table") for k in header):
            return "ltx_video_v0_9_7"
        elif has_vae and s(header, "vae.encoder.conv_out.conv.weight")[1] == 2048:
            return "ltx_video_v0_9_5"
        elif "vae.decoder.last_time_embedder.timestep_embedder.linear_1.weight" in header:
            return "ltx_video_v0_9_1"
        else:
            return "ltx_video"

    elif CHECKPOINT_KEY_NAMES["autoencoder-dc"] in header:
        encoder_key = "encoder.project_in.conv.conv.bias"
        decoder_key = "decoder.project_in.main.conv.weight"
        if CHECKPOINT_KEY_NAMES["autoencoder-dc-sana"] in header:
            return "autoencoder_dc_f32c32_sana"
        elif s(header, encoder_key)[-1] == 64 and s(header, decoder_key)[1] == 32:
            return "autoencoder_dc_f32c32"
        elif s(header, encoder_key)[-1] == 64 and s(header, decoder_key)[1] == 128:
            return "autoencoder_dc_f64c128"
        else:
            return "autoencoder_dc_f128c512"

    elif has_any_key(header, CHECKPOINT_KEY_NAMES["mochi-1-preview"]):
        return "mochi_v1_preview"

    elif CHECKPOINT_KEY_NAMES["hunyuan-video"] in header:
        return "hunyuan_video"

    elif has_all_keys(header, CHECKPOINT_KEY_NAMES["auraflow"]):
        return "auraflow"

    elif (
        CHECKPOINT_KEY_NAMES["instruct-pix2pix"] in header
        and s(header, CHECKPOINT_KEY_NAMES["instruct-pix2pix"])[1] == 8
    ):
        return "instruct_pix2pix"

    elif has_any_key(header, CHECKPOINT_KEY_NAMES["lumina2"]):
        return "lumina_v2"

    elif has_any_key(header, CHECKPOINT_KEY_NAMES["sana"]):
        return "sana"

    elif has_any_key(header, CHECKPOINT_KEY_NAMES["wan"]):
        target_key = (
            "model.diffusion_model.patch_embedding.weight"
            if "model.diffusion_model.patch_embedding.weight" in header
            else "patch_embedding.weight"
        )
        if CHECKPOINT_KEY_NAMES["wan_vace"] in header:
            if s(header, target_key)[0] == 1536:
                return "wan_vace_1_3b"
            elif s(header, target_key)[0] == 5120:
                return "wan_vace_14b"
        elif s(header, target_key)[0] == 1536:
            return "wan_t2v_1_3b"
        elif s(header, target_key)[0] == 5120 and s(header, target_key)[1] == 16:
            return "wan_t2v_14b"
        else:
            return "wan_i2v_14b"

    elif CHECKPOINT_KEY_NAMES["wan_vae"] in header:
        return "wan_t2v_14b"

    elif CHECKPOINT_KEY_NAMES["hidream"] in header:
        return "hidream"

    elif has_all_keys(header, CHECKPOINT_KEY_NAMES["cosmos-1.0"]):
        shape = s(header, CHECKPOINT_KEY_NAMES["cosmos-1.0"][0])
        if shape[1] == 68:
            return "cosmos_v1_t2w_7b" if shape[0] == 4096 else "cosmos_v1_t2w_14b"
        elif shape[1] == 72:
            return "cosmos_v1_v2w_7b" if shape[0] == 4096 else "cosmos_v1_v2w_14b"
    elif has_all_keys(header, CHECKPOINT_KEY_NAMES["cosmos-2.0"]):
        shape = s(header, CHECKPOINT_KEY_NAMES["cosmos-2.0"][0])
        if shape[1] == 68:
            return "cosmos_v2_t2i_2b" if shape[0] == 2048 else "cosmos_v2_t2i_14b"
        elif shape[1] == 72:
            return "cosmos_v2_v2w_2b" if shape[0] == 2048 else "cosmos_v2_v2w_14b"

    return "sd_v1_base"


def identify_model_type(path):
    if path.lower().endswith(".safetensors") or path.lower().endswith(".sft"):
        header = read_safetensors_header(path)
    elif path.lower().endswith(".gguf"):
        header = read_gguf_header(path)
    else:
        print(f"Unsupported file type: {path}")
        return
    return infer_diffusers_model_type(header)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python model_identifier.py <model.safetensors>")
        sys.exit(1)
    print(identify_model_type(sys.argv[1]))
