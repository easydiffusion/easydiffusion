import create from "zustand";
import produce from "immer";
import { useRandomSeed } from "../utils";

import { ImageRequest } from "./imageCreateStore";

interface ImageQueueState {
  images: ImageRequest[];
  completedImageIds: string[];
  addNewImage: (id: string, imgRec: ImageRequest) => void;
  hasQueuedImages: () => boolean;
  firstInQueue: () => ImageRequest | {};
  removeFirstInQueue: () => void;
  clearCachedIds: () => void;
}

export const useImageQueue = create<ImageQueueState>((set, get) => ({
  images: [],
  completedImageIds: [],
  // use produce to make sure we don't mutate state
  addNewImage: (id: string, imgRec: ImageRequest, isRandom = false) => {
    set(
      produce((state) => {
        let { seed } = imgRec;
        if (isRandom) {
          seed = useRandomSeed();
        }
        state.images.push({ id, options: { ...imgRec, seed } });
      })
    );
  },

  hasQueuedImages: () => {
    return get().images.length > 0;
  },

  firstInQueue: () => {
    let first: ImageRequest | {} = get().images[0]
    first = void 0 !== first ? first : {};
    return first;
  },

  removeFirstInQueue: () => {
    set(
      produce((state) => {
        const image = state.images.shift();
        state.completedImageIds.push(image.id);
      })
    );
  },

  clearCachedIds: () => {
    set(
      produce((state) => {
        state.completedImageIds = [];
      })
    );
  },
}));
