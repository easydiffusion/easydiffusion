import create from "zustand";
import produce from "immer";

import { ImageRequest } from "../api";

export enum QueueStatus {
  pending = "pending",
  processing = "processing",
  complete = "complete",
  paused = "paused",
  error = "error",
}

export interface QueuedRequest {
  id: string;
  options: ImageRequest;
  status: QueueStatus[keyof QueueStatus];
  //"pending" | "processing" | "complete" | "error";
}

interface RequestQueueState {
  requests: QueuedRequest[];
  addtoQueue: (id: string, imgRec: ImageRequest) => void;
  pendingRequests: () => QueuedRequest[];
  hasPendingQueue: () => boolean;
  hasAnyQueue: () => boolean;
  firstInQueue: () => QueuedRequest;
  updateStatus: (id: string, status: QueueStatus[keyof QueueStatus]) => void;
  sendPendingToTop: (id: string) => void;
  removeItem: (id: string) => void;
  removeCompleted: () => void;
  removeErrored: () => void;
  clearQueue: () => void;

}

export const useRequestQueue = create<RequestQueueState>((set, get) => ({
  requests: [],
  // use produce to make sure we don't mutate state
  addtoQueue: (id: string, imgRec: ImageRequest) => {
    set(
      produce((state) => {
        const item: QueuedRequest = { id, options: imgRec, status: QueueStatus.pending };
        state.requests.push(item);
      })
    );
  },

  pendingRequests: () => {
    return get().requests.filter((item) => item.status === QueueStatus.pending);
  },

  hasPendingQueue: () => {
    return get().pendingRequests().length > 0;
  },

  hasAnyQueue: () => {
    return get().requests.length > 0;
  },

  firstInQueue: () => {
    const pending = get().pendingRequests()[0];

    if (pending === undefined) {
      // eslint-disable-next-line @typescript-eslint/consistent-type-assertions
      const temp: QueuedRequest = { id: "", options: ({} as ImageRequest), status: QueueStatus.pending };
      return temp;
    }
    return pending;
  },

  updateStatus: (id: string, status: QueueStatus[keyof QueueStatus]) => {
    set(
      produce((state) => {
        const item = state.requests.find((item: QueuedRequest) => item.id === id);
        if (void 0 !== item) {
          item.status = status;
        }
      })
    );
  },

  sendPendingToTop: (id: string) => {
    set(
      produce((state) => {
        const item = state.requests.find((item: QueuedRequest) => item.id === id);
        if (void 0 !== item) {
          const index = state.requests.indexOf(item);
          // insert infront of the pending requests
          const pending = get().pendingRequests();
          const pendingIndex = state.requests.indexOf(pending[0]);
          console.log()
          state.requests.splice(index, 1);
          state.requests.splice(pendingIndex, 0, item);
        }
      })
    );
  },

  removeItem: (id: string) => {
    set(
      produce((state) => {
        const index = state.requests.findIndex((item: QueuedRequest) => item.id === id);
        if (index > -1) {
          state.requests.splice(index, 1);
        }
      })
    );
  },

  removeCompleted: () => {
    set(
      produce((state) => {
        const completed = state.requests.filter((item: QueuedRequest) => item.status === QueueStatus.complete);
        completed.forEach((item: QueuedRequest) => {
          const index = state.requests.indexOf(item);
          state.requests.splice(index, 1);
        });
      })
    );
  },

  removeErrored: () => {
    set(
      produce((state) => {
        const errored = state.requests.filter((item: QueuedRequest) => item.status === QueueStatus.error);
        errored.forEach((item: QueuedRequest) => {
          const index = state.requests.indexOf(item);
          state.requests.splice(index, 1);
        });
      })
    );
  },


  clearQueue: () => {
    set(
      produce((state) => {
        state.requests = [];
      })
    );
  },
}));
