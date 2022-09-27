/* eslint-disable @typescript-eslint/restrict-plus-operands */
import create from "zustand";
import produce from "immer";

import { ImageRequest } from "./imageCreateStore";

export interface CompletedImagesType {
  id?: string;
  data: string | undefined;
  info: ImageRequest;
}


interface ImageDisplayState {
  // imageOptions: Map<string, any>;
  images: CompletedImagesType[]
  currentImage: CompletedImagesType | null
  updateDisplay: (ImageData: string, imageOptions: any) => void;
  setCurrentImage: (image: CompletedImagesType) => void;
  clearDisplay: () => void;

  // getCurrentImage: () => {};
}

export const useImageDisplay = create<ImageDisplayState>((set, get) => ({
  imageMap: new Map<string, any>(),
  images: [],
  currentImage: null,
  // use produce to make sure we don't mutate state
  // imageOptions: any
  updateDisplay: (ImageData: string, imageOptions) => {
    set(
      produce((state) => {
        // options: imageOptions
        // state.currentImage = { display: ImageData, imageOptions };
        // imageOptions
        state.images.unshift({ data: ImageData, info: imageOptions });
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
