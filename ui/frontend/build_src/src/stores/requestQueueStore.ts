import create from "zustand";
import produce from "immer";

import { ImageRequest } from "@api/api.d";

export enum QueueStatus {
  pending = "pending",
  processing = "processing",
  complete = "complete",
  paused = "paused",
  error = "error",
}

export interface QueuedRequest {
  batchId: string;
  options: ImageRequest;
  status: QueueStatus[keyof QueueStatus];
  //"pending" | "processing" | "complete" | "error";
}

interface RequestQueueState {
  requests: QueuedRequest[];
  addtoQueue: (batchId: string, imgRec: ImageRequest) => void;
  pendingRequests: () => QueuedRequest[];
  hasPendingQueue: () => boolean;
  hasAnyQueue: () => boolean;
  firstInQueue: () => QueuedRequest;
  updateStatus: (batchId: string, status: QueueStatus[keyof QueueStatus]) => void;
  sendPendingToTop: (batchId: string) => void;
  removeItem: (batchId: string) => void;
  removeCompleted: () => void;
  removeErrored: () => void;
  clearQueue: () => void;
}

export const useRequestQueue = create<RequestQueueState>((set, get) => ({
  requests: [],
  // use produce to make sure we don't mutate state
  addtoQueue: (batchId: string, imgRec: ImageRequest) => {
    set(
      produce((state) => {
        const item: QueuedRequest = { batchId, options: imgRec, status: QueueStatus.pending };
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
      const temp: QueuedRequest = { batchId: "", options: ({} as ImageRequest), status: QueueStatus.pending };
      return temp;
    }
    return pending;
  },

  updateStatus: (batchId: string, status: QueueStatus[keyof QueueStatus]) => {
    set(
      produce((state) => {
        const item = state.requests.find((item: QueuedRequest) => item.batchId === batchId);
        if (void 0 !== item) {
          item.status = status;
        }
      })
    );
  },

  sendPendingToTop: (batchId: string) => {
    set(
      produce((state) => {
        const item = state.requests.find((item: QueuedRequest) => item.batchId === batchId);

        if (void 0 !== item) {
          // remove from current position
          const index = state.requests.indexOf(item);
          state.requests.splice(index, 1);

          // find the first available stop and insert it there
          for (let i = 0; i < state.requests.length; i++) {
            const curStatus = state.requests[i].status;

            // skip over any items that are not pending or paused
            if (curStatus === QueueStatus.processing) {
              continue;
            }
            if (curStatus === QueueStatus.complete) {
              continue;
            }
            if (curStatus === QueueStatus.error) {
              continue;
            }

            // insert infront of any pending or paused items
            state.requests.splice(i, 0, item);
            break;
          }
        }
      })
    );
  },

  removeItem: (batchId: string) => {
    set(
      produce((state) => {
        const index = state.requests.findIndex((item: QueuedRequest) => item.batchId === batchId);
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
