import create from "zustand";
import produce from "immer";

// import { devtools } from 'zustand/middleware'

interface ImageDisplayState {
  imageOptions: Map<string, any>;
  currentImage: object | null;
  addNewImage: (ImageData: string, imageOptions: any) => void;
}

export const useImageDisplay = create<ImageDisplayState>((set) => ({
  imageOptions: new Map<string, any>(),
  currentImage: null,
  // use produce to make sure we don't mutate state
  addNewImage: (ImageData: string, imageOptions: any) => {
    set(
      produce((state) => {
        state.currentImage = { display: ImageData, options: imageOptions };
        state.images.set(ImageData, imageOptions);
      })
    );
  },
}));
