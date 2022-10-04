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
  completed: boolean;
  data: imageDataObject[] | undefined;
  info: ImageRequest;
}

interface createdMediaState {
  createdMediaList: createdMediaObject[];
  createdMediaRecords: Record<string, createdMediaObject[]>;
  makeSpace: (batchId: string, req: ImageRequest) => void;
  addCreatedMedia: (batchId: string, data: imageDataObject) => void;
  addCreatedMediaRecord: (batchId: string, seed: string, data: imageDataObject) => void;
  getCreatedMedia: (batchId: string) => createdMediaObject | undefined;
  removeFailedMedia: (batchId: string) => void;

};

export const useCreatedMedia = create<createdMediaState>((set, get) => ({
  //Array<createdMedia>(),
  createdMediaList: [],
  createdMediaRecords: {},
  makeSpace: (batchId: string, req: ImageRequest) => {
    set(
      produce((state) => {
        const item: createdMediaObject = {
          batchId,
          completed: false,
          data: [],
          info: req,
        };
        state.createdMediaList.unshift(item);

      })
    );
  },
  addCreatedMedia: (batchId: string, data: imageDataObject) => {
    set(
      produce((state) => {
        const media = state.createdMediaList.find((item: createdMediaObject) => item.batchId === batchId);
        if (void 0 !== media) {
          media.data.push(data);
        }
      })
    );
  },

  addCreatedMediaRecord: (batchId: string, seed: string, data: imageDataObject) => {
    set(
      produce((state) => {
        const media = state.createdMediaList.find((item: createdMediaObject) => item.batchId === batchId);
        if (void 0 !== media) {
          if (void 0 === state.createdMediaRecords[seed]) {
            state.createdMediaRecords[seed] = [];
          }
          state.createdMediaRecords[seed].push(media);
        }
      })
    );
  },

  getCreatedMedia: (batchId: string) => {
    const item = get().createdMediaList.find((item: createdMediaObject) => item.batchId === batchId);
    if (void 0 !== item) {
      return item;
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