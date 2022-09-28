/**
 * basic server health
 */

import type { SAMPLER_OPTIONS } from "../stores/imageCreateStore";

// when we are on dev we want to specifiy 9000 as the port for the backend
// when we are on prod we want be realtive to the current url
export const API_URL = import.meta.env.DEV ? "http://localhost:9000" : "";

export const HEALTH_PING_INTERVAL = 5000; // 5 seconds
export const healthPing = async () => {
  const pingUrl = `${API_URL}/ping`;
  const response = await fetch(pingUrl);
  const data = await response.json();
  return data;
};

/**
 * the local list of modifications
 */
export const loadModifications = async () => {
  const response = await fetch(`${API_URL}/modifiers.json`);
  const data = await response.json();
  return data;
};

export const getSaveDirectory = async () => {
  const response = await fetch(`${API_URL}/output_dir`);
  const data = await response.json();
  return data[0];
};

export const KEY_CONFIG = "config";
export const getConfig = async () => {
  const response = await fetch(`${API_URL}/app_config`);
  const data = await response.json();
  return data;
};

export const KEY_TOGGLE_CONFIG = "toggle_config";
export const toggleBetaConfig = async (branch: string) => {
  const response = await fetch(`${API_URL}/app_config`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      update_branch: branch,
    }),
  });
  const data = await response.json();
  return data;
};

/**
 * post a new request for an image
 */
// TODO; put hese some place better
export interface ImageRequest {
  session_id: string;
  prompt: string;
  seed: number;
  num_outputs: number;
  num_inference_steps: number;
  guidance_scale: number;
  width:
  | 128
  | 192
  | 256
  | 320
  | 384
  | 448
  | 512
  | 576
  | 640
  | 704
  | 768
  | 832
  | 896
  | 960
  | 1024;
  height:
  | 128
  | 192
  | 256
  | 320
  | 384
  | 448
  | 512
  | 576
  | 640
  | 704
  | 768
  | 832
  | 896
  | 960
  | 1024;
  // allow_nsfw: boolean
  turbo: boolean;
  use_cpu: boolean;
  use_full_precision: boolean;
  save_to_disk_path: null | string;
  use_face_correction: null | "GFPGANv1.3";
  use_upscale: null | "RealESRGAN_x4plus" | "RealESRGAN_x4plus_anime_6B" | "";
  show_only_filtered_image: boolean;
  init_image: undefined | string;
  prompt_strength: undefined | number;
  mask: undefined | string;
  sampler: typeof SAMPLER_OPTIONS[number];
  stream_progress_updates: true;
  stream_image_progress: boolean;

}

export interface ImageOutput {
  data: string;
  path_abs: string | null;
  seed: number;
}

export interface ImageReturnType {
  output: ImageOutput[];
  request: ImageRequest;
  status: string;
}

export const MakeImageKey = "MakeImage";
export const doMakeImage = async (reqBody: ImageRequest) => {
  const res = await fetch(`${API_URL}/image`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(reqBody),
  });
  return res;
};

export const doStopImage = async () => {

  console.log("stopping image");
  const response = await fetch(`${API_URL}/image/stop`);
  console.log("stopping image response", response);
  const data = await response.json();
  console.log("stopping image data", data);
  return data;

  //   try {
  //     let res = await fetch('/image/stop')
  // } catch (e) {
  //     console.log(e)
  // }
};