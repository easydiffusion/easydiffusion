import create from "zustand";
import produce from "immer";

export const FetchingStates = {
  IDLE: "IDLE",
  FETCHING: "FETCHING",
  PROGRESSING: "PROGRESSING",
  SUCCEEDED: "SUCCEEDED",
  COMPLETE: "COMPLETE",
  ERROR: "ERROR",
} as const;

interface ImageFetchingState {
  status: typeof FetchingStates[keyof typeof FetchingStates];
  step: number;
  totalSteps: number;
  data: string;
  timeStarted: Date;
  timeNow: Date;
  appendData: (data: string) => void;
  reset: () => void;
  setStatus: (status: typeof FetchingStates[keyof typeof FetchingStates]) => void;
  setStep: (step: number) => void;
  setTotalSteps: (totalSteps: number) => void;
  setStartTime: () => void;
  setNowTime: () => void;
  resetForFetching: () => void;
}

export const useImageFetching = create<ImageFetchingState>((set) => ({
  status: FetchingStates.IDLE,
  step: 0,
  totalSteps: 0,
  data: '',
  progressImages: [],
  timeStarted: new Date(),
  timeNow: new Date(),
  // use produce to make sure we don't mutate state
  appendData: (data: string) => {
    set(
      produce((state: ImageFetchingState) => {
        // eslint-disable-next-line @typescript-eslint/restrict-plus-operands
        state.data += data;
      })
    );
  },
  reset: () => {
    set(
      produce((state: ImageFetchingState) => {
        state.status = FetchingStates.IDLE;
        state.step = 0;
        state.totalSteps = 0;
        state.data = '';
      })
    );
  },
  setStatus: (status: typeof FetchingStates[keyof typeof FetchingStates]) => {
    set(
      produce((state: ImageFetchingState) => {
        state.status = status;
      })
    );
  },
  setStep: (step: number) => {
    set(
      produce((state: ImageFetchingState) => {
        state.step = step;
      })
    );
  },
  setTotalSteps: (totalSteps: number) => {
    set(
      produce((state: ImageFetchingState) => {
        state.totalSteps = totalSteps;
      })
    );
  },

  setStartTime: () => {
    set(
      produce((state: ImageFetchingState) => {
        state.timeStarted = new Date();
      })
    );
  },
  setNowTime: () => {
    set(
      produce((state: ImageFetchingState) => {
        state.timeNow = new Date();
      })
    );
  },
  resetForFetching: () => {
    set(
      produce((state: ImageFetchingState) => {
        state.status = FetchingStates.FETCHING;
        state.step = 0;
        state.totalSteps = 0;
        state.timeNow = new Date();
        state.timeStarted = new Date();
      })
    );
  }
}));
