/* eslint-disable @typescript-eslint/strict-boolean-expressions */
import create from "zustand";
import produce from "immer";

import { ImageRequest } from "../api/api.d";

// same as progressDataObject
export interface imageDataObject {
  id: string;
  data: string;
}

interface createdMediaObject {
  batchId: string;
  seedId: number;
  data: imageDataObject[];
  info: ImageRequest;
}

interface createdMediaState {
  createdMediaList: createdMediaObject[];
  addCreatedMedia: (batchId: string, seed: number, info: ImageRequest, data: imageDataObject) => void;
  getCreatedMedia: (batchId: string, seed: number) => createdMediaObject | undefined;
  removeFailedMedia: (batchId: string) => void;
};

export const useCreatedMedia = create<createdMediaState>((set, get) => ({
  //Array<createdMedia>(),
  createdMediaList: [],
  addCreatedMedia: (batchId: string, seed: number, info: ImageRequest, data: imageDataObject) => {
    set(produce((state) => {
      const batch = state.createdMediaList.find((batch: { batchId: string; }) => batch.batchId === batchId);
      if (batch) {
        batch.data.push(data);
      } else {
        state.createdMediaList.push({
          batchId,
          seedId: seed,
          data: [data],
          info,
        });
      }
    }));
  },

  getCreatedMedia: (batchId: string, seed: number) => {
    const batch = get().createdMediaList.find((batch) => batch.batchId === batchId && batch.seedId === seed);
    if (batch) {
      return batch;
    }
    return undefined;
  },

  removeFailedMedia: (batchId: string) => {
    set(
      produce((state) => {
        const index = state.createdMediaList.findIndex((media: createdMediaObject) => media.batchId === batchId);
        if (index !== -1) {
          state.createdMedia.splice(index, 1);
        }
      })
    );
  }

}));