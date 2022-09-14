import create from 'zustand';
import produce from 'immer';
import { useRandomSeed } from '../utils';

import { ImageRequest } from './imageCreateStore';

interface ImageQueueState {
  images : ImageRequest[];
  completedImageIds: string[];
  addNewImage: (id:string, imgRec: ImageRequest) => void
  hasQueuedImages: () => boolean;
  firstInQueue: () => ImageRequest | [];
  removeFirstInQueue: () => void;
}

// figure out why TS is complaining about this
// @ts-ignore
export const useImageQueue = create<ImageQueueState>((set, get) => ({
  images: new Array(),
  completedImageIds: new Array(),
  // use produce to make sure we don't mutate state
  addNewImage: (id: string, imgRec: ImageRequest, isRandom= false) => {
    set( produce((state) => {

      let { seed } = imgRec;
      if (isRandom) {
        seed = useRandomSeed();
      }
      state.images.push({ id, options: {...imgRec, seed} });
    }));
  },
  
  hasQueuedImages: () => {
    return get().images.length > 0;
  },
  firstInQueue: () => {
    return get().images[0] as ImageRequest || []; 
  },
  
  removeFirstInQueue: () => {
    set( produce((state) => {
      const image = state.images.shift();
      state.completedImageIds.push(image.id);
    }));
  }
}));
  