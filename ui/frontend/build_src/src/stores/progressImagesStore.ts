/* eslint-disable @typescript-eslint/strict-boolean-expressions */
import create from "zustand";
import produce from "immer";

// same as imageDataObject
export interface progressDataObject {
  id: string;
  data: string;
};

interface progressImageList {
  batchId: string;
  progressImageBySeed: Record<string, progressDataObject[]>;
};

interface progressImagesState {
  progressImagesManifest: progressImageList[];
  addProgressImage: (batchId: string, seed: string, imageData: progressDataObject) => void;
  getProgressImageBatch: (batchId: string) => progressImageList | null;
  getProgressImageList: (batchId: string, seed: string) => progressDataObject[];
};

export const useProgressImages = create<progressImagesState>((set, get) => ({
  progressImagesManifest: [],
  addProgressImage: (batchId: string, seed: string, imageData: progressDataObject) => {
    set(produce((state) => {
      const batch = state.progressImagesManifest.find((batch: { batchId: string; }) => batch.batchId === batchId);
      if (batch) {
        batch.progressImages.push(imageData);
        const seedRec = batch.progressImageBySeed[seed];
        if (seedRec) {
          seedRec.push(imageData);
        } else {
          batch.progressImageBySeed[seed] = [imageData];
        }
      } else {
        state.progressImagesManifest.push({
          batchId,
          progressImages: [imageData],
          progressImageBySeed: {
            [seed]: [imageData],
          },
        });
      }
    }));
  },

  getProgressImageBatch: (batchId: string) => {
    const batch = get().progressImagesManifest.find((batch: { batchId: string; }) => batch.batchId === batchId);
    if (batch) {
      return batch;
    }
    return null;
  },

  getProgressImageList: (batchId: string, seed: string) => {
    const batch = get().progressImagesManifest.find((batch) => batch.batchId === batchId);
    if (batch) {
      return batch.progressImageBySeed[seed] || [];
    }
    return [];
  },

}));