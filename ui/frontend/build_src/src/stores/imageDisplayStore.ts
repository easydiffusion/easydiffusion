import create from "zustand";
import produce from "immer";

export interface CompletedImageIds {
  batchId: string;
  imageId: string;
}

interface ImageDisplayState {
  currentImageKeys: CompletedImageIds | null
  setCurrentImage: (image: CompletedImageIds) => void;
  clearDisplay: () => void;
}

export const useImageDisplay = create<ImageDisplayState>((set, get) => ({
  currentImageKeys: null,

  setCurrentImage: (image) => {
    set(
      produce((state) => {
        state.currentImage = image;
      })
    );
  },

  clearDisplay: () => {
    set(
      produce((state) => {
        state.currentImage = null;
      })
    );
  }

}));
