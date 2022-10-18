import create from "zustand";
import produce from "immer";

export interface CompletedImageIds {
  batchId: string;
  seed: number;
  imageId?: string;
  progressId?: string;
}

interface ImageDisplayState {
  currentImageKeys: CompletedImageIds | null
  shouldDisplayWhenComplete: boolean,
  setCurrentImage: (image: CompletedImageIds) => void;
  clearDisplay: () => void;
  toggleDisplayComplete: () => void;
}

export const useImageDisplay = create<ImageDisplayState>((set, get) => ({
  currentImageKeys: null,
  shouldDisplayWhenComplete: true,
  setCurrentImage: (image) => {
    set(
      produce((state) => {
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
  },
  toggleDisplayComplete: () => {
    set(
      produce((state) => {
        // eslint-disable-next-line @typescript-eslint/strict-boolean-expressions
        state.shouldDisplayWhenComplete = !(state.shouldDisplayWhenComplete);
      })
    );
  }

}));
