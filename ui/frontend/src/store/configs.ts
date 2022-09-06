import { defineStore } from "pinia"
import { computed, ref } from "vue";
import { DEFAULT_STEPS, DEFAULT_GUIDANCE, DEFAULT_WIDTH, DEFAULT_HEIGHT } from "../constants";


export const useConfigsStore = defineStore('configs', () => {
  const seed = ref('');
  const steps = ref(`${DEFAULT_STEPS}`);
  const guidance = ref(DEFAULT_GUIDANCE);
  const width = ref(`${DEFAULT_WIDTH}`);
  const height = ref(`${DEFAULT_HEIGHT}`);

  return {
    seed,
    steps,
    guidance,
    width,
    height,
  }
})