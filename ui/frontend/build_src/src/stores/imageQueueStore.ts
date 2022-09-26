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
        const newImg = { id, options: { ...imgRec, seed } };
        console.log("addNewImage", newImg);

        state.images.push(newImg);
      })
    );
  },

  hasQueuedImages: () => {
    return get().images.length > 0;
  },

  firstInQueue: () => {
    let first: ImageRequest | {} = get().images[0];
    first = void 0 !== first ? first : {};
    return first;
  },

  removeFirstInQueue: () => {
    set(
      produce((state) => {
        console.log("removing first in queue");
        const image = state.images.shift();
        console.log("image", image);
        // state.completedImageIds.push(image.id);
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
