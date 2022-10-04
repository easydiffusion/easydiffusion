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
  progressImages: progressDataObject[];
};

interface progressImagesState {
  progressImagesManifest: progressImageList[];
  addProgressImage: (batchId: string, imageData: progressDataObject) => void;
  getProgressImages: (batchId: string) => progressDataObject[];
};

export const useProgressImages = create<progressImagesState>((set, get) => ({
  progressImagesManifest: [],
  addProgressImage: (batchId: string, imageData: progressDataObject) => {
    set(
      produce((state) => {
        const item = state.progressImagesManifest.find((item: progressImageList) => item.batchId === batchId);
        if (void 0 !== item) {
          console.log('add new item ', imageData.id);
          console.log('to batch', batchId);
          item.progressImages.unshift(imageData);
          console.log('new length ', item.progressImages.length)
        }
        else {
          console.log('add new batch', batchId);
          console.log('first item', imageData.id);
          state.progressImagesManifest.push({ batchId, progressImages: [imageData] });
        }
      })
    );
  },
  getProgressImages: (batchId: string) => {
    const item = get().progressImagesManifest.find((item: progressImageList) => item.batchId === batchId);
    if (void 0 !== item) {
      return item.progressImages;
    }

    return [];
  }
}));