import create from "zustand";
import produce from "immer";
import { useRandomSeed } from "../utils";

import { ImageRequest } from "./imageCreateStore";

interface QueueItem {
  id?: string;
  options?: ImageRequest;
  status?: "pending" | "complete" | "error";
}

interface ImageQueueState {
  images: QueueItem[];
  completedImageIds: string[];
  addNewImage: (id: string, imgRec: ImageRequest) => void;
  hasQueuedImages: () => boolean;
  firstInQueue: () => QueueItem;
  removeFirstInQueue: () => void;
  clearCachedIds: () => void;
}

export const useImageQueue = create<ImageQueueState>((set, get) => ({
  images: [],
  completedImageIds: [],
  // use produce to make sure we don't mutate state
  addNewImage: (id: string, imgRec: ImageRequest) => {
    set(
      produce((state) => {
        const item: QueueItem = { id, options: imgRec, status: "pending" };
        state.images.push(item);
      })
    );
  },

  hasQueuedImages: () => {
    return get().images.length > 0;
  },

  firstInQueue: () => {
    const { images } = get();
    if (images.length > 0) {
      return images[0];
    }
    // // cast an empty object to QueueItem
    const empty: QueueItem = {};
    return empty;

  },

  removeFirstInQueue: () => {
    set(
      produce((state) => {
        const image = state.images.shift();
        if (void 0 !== image) {
          state.completedImageIds.push(image.id);
        }
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
