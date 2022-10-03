/* eslint-disable @typescript-eslint/strict-boolean-expressions */
import create from "zustand";
import produce from "immer";
import { devtools } from "zustand/middleware";

import { useRandomSeed } from "../utils";

import { ImageRequest } from "../api";

// import {
//   FetchingStates
// } from "./imageFetchingStore";

interface imageDataObject {
  id: string;
  data: string;
}

interface createdMedia {
  id: string;
  completed: boolean;
  data: imageDataObject[] | undefined;
  positivePrompts: string[];
  negativePrompts: string[];
  info: ImageRequest;
  progressImages: string[];
}

interface createdMediaState {
  createdMedia: createdMedia[];
  makeSpace: (id: string, req: ImageRequest) => void;
  addCreatedMedia: (id: string, data: imageDataObject) => void;
  // completeCreatedMedia: (id: string) => void;
  addProgressImage: (id: string, imageData: string) => void;
  removeFailedMedia: (id: string) => void;
};

export const useCreatedMedia = create<createdMediaState>((set, get) => ({
  createdMedia: [],

  makeSpace: (id: string, req: ImageRequest) => {
    set(
      produce((state) => {
        const item: createdMedia = {
          id,
          completed: false,
          data: [],
          positivePrompts: [],
          negativePrompts: [],
          // eslint-disable-next-line @typescript-eslint/consistent-type-assertions
          info: req,
          progressImages: []
        };
        state.createdMedia.unshift(item);

      })
    );
  },
  addCreatedMedia: (id: string, data: imageDataObject) => {
    set(

      produce((state) => {
        const item = state.createdMedia.find((item: createdMedia) => item.id === id);
        if (void 0 !== item) {
          item.data.push(data);
          item.completed = true;
        }
      })
    );
  },

  // completeCreatedMedia: (id) => {
  //   set(
  //     produce((state) => {
  //       const index = state.createdMedia.findIndex((media) => media.id === id);
  //       if (index !== -1) {
  //         state.createdMedia[index].completed = true;
  //       }
  //     })
  //   );
  // },

  addProgressImage(id, imageData) {
    set(
      produce((state) => {
        const index = state.createdMedia.findIndex((media) => media.id === id);
        state.createdMedia[index].progressImages.push(imageData);
      })
    );
  },

  removeFailedMedia: (id) => {
    set(
      produce((state) => {
        const index = state.createdMedia.findIndex((media) => media.id === id);
        if (index !== -1) {
          state.createdMedia.splice(index, 1);
        }
      })
    );
  }

}));