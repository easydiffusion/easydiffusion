import create from "zustand";
import produce from "immer";
import { v4 as uuidv4 } from "uuid";

import { ImageRequest } from "@api/api";

import { promptTag } from "./store.d";

export enum QueueStatus {
  pending = "pending",
  processing = "processing",
  complete = "complete",
  paused = "paused",
  error = "error",
}

export interface QueuedPrompt {
  queueId: string;
  options: promptTag[];
}

interface PromptMatixState {

  shouldClearOnAdd: boolean;
  shouldClearOnCreate: boolean;
  potentialTags: string;
  promptsList: QueuedPrompt[];
  setPotentialTags: (tags: string) => void;
  generateTags: (isPositive: boolean) => void;
  togglePromptType: (queueId: string, tagId: string) => void;
  removePromptModifier: (queueId: string, tagId: string) => void;
  getSafeList: () => QueuedPrompt[];
  clearPromptMatrix: () => void;
}

export const usePromptMatrix = create<PromptMatixState>((set, get) => ({
  shouldClearOnAdd: true,
  shouldClearOnCreate: true,
  potentialTags: '',

  promptsList: [],

  setPotentialTags: (tags: string) => {
    set({ potentialTags: tags });
  },

  generateTags: (isPositive: boolean) => {
    set(
      produce((state) => {
        const tags = state.potentialTags.split(',');
        const tagList: promptTag[] = [];
        tags.forEach((tag) => {
          tagList.push({ id: uuidv4(), name: tag, type: isPositive ? 'positive' : 'negative' });
        });
        state.promptsList.push({ queueId: uuidv4(), options: tagList });
      })
    );

    if (get().shouldClearOnAdd) {
      set({ potentialTags: '' });
    }
  },

  togglePromptType: (queueId: string, tagId: string) => {
    set(
      produce((state) => {
        const prompt = state.promptsList.find((prompt) => prompt.queueId === queueId);
        const tag = prompt.options.find((tag) => tag.id === tagId);
        tag.type = tag.type === 'positive' ? 'negative' : 'positive';
      })
    );
  },

  removePromptModifier: (queueId: string, tagId: string) => {
    set(
      produce((state) => {
        const prompt = state.promptsList.find((prompt) => prompt.queueId === queueId);
        const tagIndex = prompt.options.findIndex((tag) => tag.id === tagId);
        prompt.options.splice(tagIndex, 1);
      })
    );
  },

  getSafeList: () => {

    // return an sinlge item with an empty list if there are no prompts
    if (get().promptsList.length === 0) {
      return [{
        queueId: uuidv4(),
        options: []
      }];
    }

    return get().promptsList;
  },

  clearPromptMatrix: () => {
    set(
      produce((state) => {
        state.prompts = [];
      })
    );
  },

}));
