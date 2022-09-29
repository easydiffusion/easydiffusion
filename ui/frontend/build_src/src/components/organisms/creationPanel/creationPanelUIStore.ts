import create from "zustand";
import produce from "immer";
import { persist } from "zustand/middleware";

export interface ImageCreationUIOptions {
  isOpenAdvancedSettings: boolean;
  isOpenAdvImprovementSettings: boolean;
  isOpenAdvPropertySettings: boolean;
  isOpenAdvWorkflowSettings: boolean;
  isOpenImageModifier: boolean;

  toggleAdvancedSettings: () => void;
  toggleAdvImprovementSettings: () => void;
  toggleAdvPropertySettings: () => void;
  toggleAdvWorkflowSettings: () => void;

  toggleImageModifier: () => void;
  // addImageModifier: (modifier: string) => void;
}

export const useCreateUI = create<ImageCreationUIOptions>(
  //@ts-expect-error
  persist(
    (set, get) => ({
      isOpenAdvancedSettings: false,
      isOpenAdvImprovementSettings: false,
      isOpenAdvPropertySettings: false,
      isOpenAdvWorkflowSettings: false,
      isOpenImageModifier: false,

      toggleAdvancedSettings: () => {
        set(
          produce((state: ImageCreationUIOptions) => {
            state.isOpenAdvancedSettings = !state.isOpenAdvancedSettings;
          })
        );
      },

      toggleAdvImprovementSettings: () => {
        set(
          produce((state: ImageCreationUIOptions) => {
            state.isOpenAdvImprovementSettings =
              !state.isOpenAdvImprovementSettings;
          })
        );
      },

      toggleAdvPropertySettings: () => {
        set(
          produce((state: ImageCreationUIOptions) => {
            state.isOpenAdvPropertySettings = !state.isOpenAdvPropertySettings;
          })
        );
      },

      toggleAdvWorkflowSettings: () => {
        set(
          produce((state: ImageCreationUIOptions) => {
            state.isOpenAdvWorkflowSettings = !state.isOpenAdvWorkflowSettings;
          })
        );
      },

      toggleImageModifier: () => {
        set(
          produce((state: ImageCreationUIOptions) => {
            state.isOpenImageModifier = !state.isOpenImageModifier;
          })
        );
      },
    }),
    {
      name: "createUI",
      // getStorage: () => localStorage,
    }
  )
);
