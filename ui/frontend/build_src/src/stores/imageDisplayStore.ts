import create from "zustand";
import produce from "immer";

import { ImageRequest } from "../api";

export interface CompletedImagesType {
  id?: string;
  data: string | undefined;
  info: ImageRequest;
}

interface ImageDisplayState {
  // imageOptions: Map<string, any>;
  images: CompletedImagesType[]
  currentImage: CompletedImagesType | null
  updateDisplay: (id: string, ImageData: string, imageOptions: any) => void;
  setCurrentImage: (image: CompletedImagesType) => void;
  clearDisplay: () => void;
}

export const useImageDisplay = create<ImageDisplayState>((set, get) => ({
  imageMap: new Map<string, any>(),
  images: [],
  currentImage: null,
  // use produce to make sure we don't mutate state
  // imageOptions: any
  updateDisplay: (id: string, ImageData: string, imageOptions) => {
    set(
      produce((state) => {
        state.currentImage = { id, display: ImageData, info: imageOptions };
        state.images.unshift({ id, data: ImageData, info: imageOptions });
        state.currentImage = state.images[0];
      })
    );
  },

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
        state.images = [];
        state.currentImage = null;
      })
    );
  }

}));
