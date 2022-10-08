export interface ImageRequest {
  session_id: string;
  prompt: string;
  negative_prompt: string;
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
  use_upscale: null | "RealESRGAN_x4plus" | "RealESRGAN_x4plus_anime_6B";
  use_stable_diffusion_model: 'sd-v1-4' | string;
  output_format: 'jpeg' | 'png',
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