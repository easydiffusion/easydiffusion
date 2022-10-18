import create from "zustand";
import produce from "immer";
import { persist } from "zustand/middleware";

export interface ImageCreationUIOptions {
  isModifyingPrompt: boolean;
  isOpenAdvancedSettings: boolean;
  isOpenAdvImprovementSettings: boolean;
  isOpenAdvPropertySettings: boolean;
  isOpenAdvWorkflowSettings: boolean;
  isOpenImageModifier: boolean;
  showQueue: boolean;

  // setIsModifyingPrompt: (isModifyingPrompt: boolean) => void;
  toggleAdvancedSettings: () => void;
  setAdvancedSettingsisOpen: (isOpen: boolean) => void;
  toggleAdvImprovementSettings: () => void;
  setAdvImprovementIsOpen: (isOpen: boolean) => void;
  toggleAdvPropertySettings: () => void;
  setAdvPropertyIsOpen: (isOpen: boolean) => void;
  toggleAdvWorkflowSettings: () => void;
  setAdvWorkflowIsOpen: (isOpen: boolean) => void;

  toggleImageModifier: () => void;
  toggleQueue: () => void;

}

export const useCreateUI = create<ImageCreationUIOptions>(
  //@ts-expect-error
  persist(
    (set, get) => ({
      isModifyingPrompt: false,
      isOpenAdvancedSettings: false,
      isOpenAdvImprovementSettings: false,
      isOpenAdvPropertySettings: false,
      isOpenAdvWorkflowSettings: false,
      isOpenImageModifier: false,
      showQueue: false,

      toggleAdvancedSettings: () => {
        set(
          produce((state: ImageCreationUIOptions) => {
            state.isOpenAdvancedSettings = !state.isOpenAdvancedSettings;
          })
        );
      },

      setAdvancedSettingsisOpen: (isOpen: boolean) => {
        set(
          produce((state: ImageCreationUIOptions) => {
            state.isOpenAdvancedSettings = isOpen;
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
      setAdvImprovementIsOpen: (isOpen: boolean) => {
        set(
          produce((state: ImageCreationUIOptions) => {
            state.isOpenAdvImprovementSettings = isOpen;
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

      setAdvPropertyIsOpen: (isOpen: boolean) => {
        set(
          produce((state: ImageCreationUIOptions) => {
            state.isOpenAdvPropertySettings = isOpen;
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

      setAdvWorkflowIsOpen: (isOpen: boolean) => {
        set(
          produce((state: ImageCreationUIOptions) => {
            state.isOpenAdvWorkflowSettings = isOpen;
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

      toggleQueue: () => {
        set(
          produce((state: ImageCreationUIOptions) => {
            state.showQueue = !state.showQueue;
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
