import create from "zustand";
import produce from "immer";

export interface CompletedImageIds {
  batchId: string;
  imageId?: string;
  progressId?: string;
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
        console.log('what is currentImageKeys', state.currentImageKeys)
        console.log("setting current image", image);
        state.currentImageKeys = image;
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
