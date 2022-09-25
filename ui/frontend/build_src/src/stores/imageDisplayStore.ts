/* eslint-disable @typescript-eslint/restrict-plus-operands */
import create from "zustand";
import produce from "immer";

interface ImageDisplayState {
  // imageOptions: Map<string, any>;
  images: object[]
  // SORT OF A HACK SOLUTION
  len: number,
  // currentImage: object | null;
  updateDisplay: (ImageData: string, imageOptions: any) => void;
  getCurrentImage: () => {};
}

export const useImageDisplay = create<ImageDisplayState>((set, get) => ({
  // imageOptions: new Map<string, any>(),
  images: [],
  len: 0,
  // currentImage: null,
  // use produce to make sure we don't mutate state
  // imageOptions: any
  updateDisplay: (ImageData: string, imageOptions) => {
    set(
      produce((state) => {
        // options: imageOptions
        // state.currentImage = { display: ImageData, imageOptions };
        // imageOptions
        state.images.push({ data: ImageData, info: imageOptions });
        state.len += 1;
        debugger
      })
    );
  },
  getCurrentImage: () => {
    debugger;
    return get().images[0];
  }


}));
