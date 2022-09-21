import create from "zustand";
import produce from "immer";
import { persist } from "zustand/middleware";

export interface ImageCreationUIOptions {
  isOpenAdvancedSettings: boolean;
  isOpenAdvImprovementSettings: boolean;
  isOpenAdvPropertySettings: boolean;
  isOpenAdvWorkflowSettings: boolean;
  isOpenAdvGPUSettings: boolean;

  isOpenImageModifier: boolean;
  imageMofidiersMap: object;

  toggleAdvancedSettings: () => void;
  toggleAdvImprovementSettings: () => void;
  toggleAdvPropertySettings: () => void;
  toggleAdvWorkflowSettings: () => void;
  toggleAdvGPUSettings: () => void;

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
      isOpenAdvGPUSettings: false,
      isOpenImageModifier: false,
      imageMofidiersMap: {},

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

      toggleAdvGPUSettings: () => {
        set(
          produce((state: ImageCreationUIOptions) => {
            state.isOpenAdvGPUSettings = !state.isOpenAdvGPUSettings;
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
