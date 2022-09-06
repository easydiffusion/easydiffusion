import { defineStore } from "pinia"
import { ref } from "vue";
import { generateImage } from "../api";
import { GeneratedImageData } from "../constants"

type ImagesStateType = {
  images: GeneratedImageData[];
}

export const useImageStore = defineStore('image', () => {
  const images = ref<GeneratedImageData[]>([]);

  const addImage = (image: GeneratedImageData) => {
    images.value.unshift(image);
  }
  const removeImage = (id: string) => {
    images.value = images.value.filter(image => image.id !== id);
  }

  return {
    images,
    addImage,
    removeImage
  }
})