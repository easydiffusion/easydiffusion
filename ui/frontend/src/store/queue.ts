import { API_URL, Prompt } from './../constants';
import { defineStore, storeToRefs } from "pinia"
import { computed, nextTick, ref } from "vue";
import { generateImage } from "../api";
import { GeneratedImageData, PayloadToQueueImage } from "../constants"
import { generateRandomSeed, generateUUID, playSound } from "../functions";
import { useConfigsStore } from "./configs";
import { useImageStore } from "./images";

type ConfigParams = {
  seed: string;
  steps: string;
  guidance: string;
  width: string;
  height: string;
}
const buildMergeImgWithConfigs = (item: PayloadToQueueImage, configs: ConfigParams): GeneratedImageData => {
  const promptClone = {...item.prompt};
  return {
    ...item,
    prompt: promptClone,
    seed: configs.seed,
    steps: configs.steps,
    guidance: configs.guidance,
    width: configs.width,
    height: configs.height,
  }
}

export const useQueueStore = defineStore('queue', () => {
  const imagesStore = useImageStore();
  const configsStore = useConfigsStore();
  const { guidance, height, seed, steps, width } = storeToRefs(configsStore);

  const items = ref<Array<PayloadToQueueImage>>([]);

  const first = computed<PayloadToQueueImage | undefined>(() => items.value?.[0]);
  const running = computed(() => !!first.value?.startedAt);

  const addToQueue = async (prompt: Prompt, amount: number = 1) => {
    const text = prompt.text.trim();
    prompt.text = text;

    const modifiersText = prompt.modifiers.join(", ");

    const finalPrompt = `${text}${modifiersText ? `, ${modifiersText}` : ""}`;
    prompt.finalPrompt = finalPrompt;

    for await (let i of Array(amount).keys()) {
      const payload: PayloadToQueueImage = {
        id: generateUUID(),
        prompt: prompt,
        startedAt: 0,
        elapsedMs: 0,
        createdAt: 0,
        queuedAt: Date.now(),
      };
      items.value.push(payload);
      await nextTick();
    }
  }

  const removeFromQueue = (id: string) => {
    items.value = items.value.filter(item => item.id !== id);
  }

  const execute = async () => {
    if (!first.value || running.value) return;

    first.value.startedAt = Date.now();
    const configParams: ConfigParams = {
      seed: seed.value || `${generateRandomSeed()}`,
      width: width.value,
      height: height.value,
      steps: steps.value,
      guidance: `${guidance.value}`,
    };

    const imageFilename = await generateImage(buildMergeImgWithConfigs(first.value, configParams));
    console.log('imageFilename', imageFilename);
    first.value.imageUrl = `${API_URL}/ai-images/data/file/${imageFilename}`;
    first.value.imageFilename = imageFilename;
    first.value.createdAt = Date.now();
    first.value.elapsedMs = first.value.createdAt - first.value.startedAt;

    const imageToAdd = items.value.shift() as PayloadToQueueImage;
    const imageToAddWithConfig = buildMergeImgWithConfigs(imageToAdd, configParams);
    if (!imageToAdd) return;
    imagesStore.addImage(imageToAddWithConfig);
    playSound(!items.value.length ? 'done' : 'one-image');
  }

  return {
    items,
    running,
    first,
    addToQueue,
    removeFromQueue,
    execute,
  }
})