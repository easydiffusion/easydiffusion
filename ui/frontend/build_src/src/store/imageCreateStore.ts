import create from 'zustand';
import produce from 'immer';
import { devtools } from 'zustand/middleware'

import { useRandomSeed } from '../utils';

export type ImageCreationUiOptions = {
  advancedSettingsIsOpen: boolean;
  imageModifierIsOpen: boolean;
  isCheckedUseUpscaling: boolean;
  isCheckUseFaceCorrection: boolean;
  isUseRandomSeed: boolean;
  isUseAutoSave: boolean;
  isSoundEnabled: boolean;
}

export type ImageRequest = {
  prompt: string;
  seed: number;
  num_outputs: number;
  num_inference_steps: number;
  guidance_scale: number
  width: 128 | 192 | 256 | 320 | 384 | 448 | 512 | 576 | 640 | 704 | 768 | 832 | 896 | 960 | 1024;
  height: 128 | 192 | 256 | 320 | 384 | 448 | 512 | 576 | 640 | 704 | 768 | 832 | 896 | 960 | 1024;
  // allow_nsfw: boolean;
  turbo: boolean;
  use_cpu: boolean;
  use_full_precision: boolean;
  save_to_disk_path: null | string;
  use_face_correction: null | 'GFPGANv1.3';
  use_upscale: null| 'RealESRGAN_x4plus' | 'RealESRGAN_x4plus_anime_6B';
  show_only_filtered_image: boolean;
  init_image: undefined | string;
  prompt_strength: undefined | number;
 };

interface ImageCreateState {
  parallelCount: number;
  requestOptions: ImageRequest;
  tags: string[];

  setParallelCount: (count: number) => void;
  setRequestOptions: (key: keyof ImageRequest, value: any) => void;
  getValueForRequestKey: (key: keyof ImageRequest) => any;

  toggleTag: (tag: string) => void;
  hasTag: (tag: string) => boolean;
  selectedTags:() => string[]
  builtRequest: () => ImageRequest;

  uiOptions: ImageCreationUiOptions;
  toggleAdvancedSettingsIsOpen: () => void;
  toggleImageModifiersIsOpen: () => void;
  toggleUseUpscaling: () => void;
  isUsingUpscaling: () => boolean;
  toggleUseFaceCorrection: () => void;
  isUsingFaceCorrection: () => boolean;
  toggleUseRandomSeed: () => void;
  isRandomSeed: () => boolean; 
  toggleUseAutoSave: () => void;
  isUseAutoSave: () => boolean;
  toggleSoundEnabled: () => void;
  isSoundEnabled: () => boolean;
}

// devtools breaks TS
// @ts-ignore
export const useImageCreate = create<ImageCreateState>(devtools((set, get) => ({
  
  parallelCount: 1,

  requestOptions:{
    prompt: 'a photograph of an astronaut riding a horse',
    seed: useRandomSeed(),
    num_outputs: 1,
    num_inference_steps: 50,
    guidance_scale: 7.5,
    width: 512,
    height: 512,
    prompt_strength: 0.8,
    // allow_nsfw: false,
    turbo: true,
    use_cpu: false,
    use_full_precision: true,
    save_to_disk_path: 'null',
    use_face_correction: null,
    use_upscale: 'RealESRGAN_x4plus',
    show_only_filtered_image: false,
  } as ImageRequest,

  tags: [] as string[],

  setParallelCount: (count: number) => set(produce((state) => {
    state.parallelCount = count;
  })),

  setRequestOptions: (key: keyof ImageRequest, value: any) => {
    set( produce((state) => {
      state.requestOptions[key] = value;
    }))
  },

  getValueForRequestKey: (key: keyof ImageRequest) => {
    return get().requestOptions[key];
  },
  
  toggleTag: (tag: string) => {
    set( produce((state) => {
      const index = state.tags.indexOf(tag);
      if (index > -1) {
        state.tags.splice(index, 1);
      } else {
        state.tags.push(tag);
      }
    }))
  },

  hasTag: (tag:string) => {
    return get().tags.indexOf(tag) > -1; 
  },

  selectedTags: () => {
    return get().tags;
  },

  // the request body to send to the server
  // this is a computed value, just adding the tags to the request
  builtRequest: () => {

    const state = get();
    const requestOptions = state.requestOptions;
    const tags = state.tags;  

    // join all the tags with a comma and add it to the prompt
    const prompt = `${requestOptions.prompt} ${tags.join(',')}`;

    const request = {
      ...requestOptions,
      prompt
    }
    // if we arent using auto save clear the save path
    if(!state.uiOptions.isUseAutoSave){
      // maybe this is "None" ?
      // TODO check this
      request.save_to_disk_path = null;
    }
    // if we arent using face correction clear the face correction
    if(!state.uiOptions.isCheckUseFaceCorrection){
      request.use_face_correction = null;
    }
    // if we arent using upscaling clear the upscaling
    if(!state.uiOptions.isCheckedUseUpscaling){
      request.use_upscale = null;
    }

    return request;
  },

  uiOptions: {
    // TODO proper persistence of all UI / user settings centrally somewhere?
    // localStorage.getItem('ui:advancedSettingsIsOpen') === 'true',
    advancedSettingsIsOpen:false,
    imageModifierIsOpen: false,
    isCheckedUseUpscaling: false,
    isCheckUseFaceCorrection: true,
    isUseRandomSeed: true,
    isUseAutoSave: false,
    isSoundEnabled: true,
  },

  toggleAdvancedSettingsIsOpen: () => {
    set( produce((state) => {
      state.uiOptions.advancedSettingsIsOpen = !state.uiOptions.advancedSettingsIsOpen;
      localStorage.setItem('ui:advancedSettingsIsOpen', state.uiOptions.advancedSettingsIsOpen);
    }))
  },

  toggleImageModifiersIsOpen: () => {
    set( produce((state) => {
      state.uiOptions.imageModifierIsOpen = !state.uiOptions.imageModifierIsOpen;
      localStorage.setItem('ui:imageModifierIsOpen', state.uiOptions.imageModifierIsOpen);
    }))
  },

  toggleUseUpscaling: () => {
    set( produce((state) => {
      state.uiOptions.isCheckedUseUpscaling = !state.uiOptions.isCheckedUseUpscaling;
      state.requestOptions.use_upscale = state.uiOptions.isCheckedUseUpscaling ? 'RealESRGAN_x4plus' : null;
      localStorage.setItem('ui:isCheckedUseUpscaling', state.uiOptions.isCheckedUseUpscaling);
    }))
  },

  isUsingUpscaling: () => {
    return get().uiOptions.isCheckedUseUpscaling;
  },

  toggleUseFaceCorrection: () => {
    set( produce((state) => {
      state.uiOptions.isCheckUseFaceCorrection = !state.uiOptions.isCheckUseFaceCorrection;
      state.use_face_correction = state.uiOptions.isCheckUseFaceCorrection ? 'GFPGANv1.3' : null;
      localStorage.setItem('ui:isCheckUseFaceCorrection', state.uiOptions.isCheckUseFaceCorrection);
    }))
  },

  isUsingFaceCorrection: () => {
    return get().uiOptions.isCheckUseFaceCorrection;
  },


  toggleUseRandomSeed: () => {
    set( produce((state) => {
      state.uiOptions.isUseRandomSeed = !state.uiOptions.isUseRandomSeed;
      state.requestOptions.seed = state.uiOptions.isUseRandomSeed ? useRandomSeed() : state.requestOptions.seed;
      localStorage.setItem('ui:isUseRandomSeed', state.uiOptions.isUseRandomSeed);
    }))
  },

  isRandomSeed: () => {
    return get().uiOptions.isUseRandomSeed;
  },

  toggleUseAutoSave: () => {
    //isUseAutoSave
    //save_to_disk_path
    set( produce((state) => {
      state.uiOptions.isUseAutoSave = !state.uiOptions.isUseAutoSave;
      localStorage.setItem('ui:isUseAutoSave', state.uiOptions.isUseAutoSave);
    }))
  },

  isUseAutoSave: () => {
    return get().uiOptions.isUseAutoSave;
  },

  toggleSoundEnabled: () => {
    set( produce((state) => {
      state.uiOptions.isSoundEnabled = !state.uiOptions.isSoundEnabled;
      //localStorage.setItem('ui:isSoundEnabled', state.uiOptions.isSoundEnabled);
    }))
  },

  isSoundEnabled: () => {
    return get().uiOptions.isSoundEnabled;
  },

})));


