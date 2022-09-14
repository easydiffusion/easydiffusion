import create from 'zustand';
import produce from 'immer';
import { useRandomSeed } from '../utils';

import { imageOptions } from './imageCreateStore';

interface ImageQueueState {
  images : imageOptions[];
  completedImageIds: string[];
  addNewImage: (id:string, imageOptions: imageOptions) => void
  hasQueuedImages: () => boolean;
  firstInQueue: () => imageOptions | [];
  removeFirstInQueue: () => void;
}

// figure out why TS is complaining about this
// @ts-ignore
export const useImageQueue = create<ImageQueueState>((set, get) => ({
  images: new Array(),
  completedImageIds: new Array(),
  // use produce to make sure we don't mutate state
  addNewImage: (id: string, imageOptions: any) => {
    set( produce((state) => {

      let { seed } = imageOptions;
      if (imageOptions.isSeedRandom) {
        seed = useRandomSeed();
      }

      state.images.push({ id, options: {...imageOptions, seed} });
    }));
  },
  
  hasQueuedImages: () => {
    return get().images.length > 0;
  },
  firstInQueue: () => {
    return get().images[0] as imageOptions || []; 
  },
  
  removeFirstInQueue: () => {
    set( produce((state) => {
      const image = state.images.shift();
      state.completedImageIds.push(image.id);
    }));
  }
}));
  