import { useIntervalFn } from "@vueuse/core";
import { onMounted, ref } from "vue";
import { checkServer } from "./api";
import { API_GENERATE_HEALTH, SoundName, SOUNDS_MAP } from "./constants";
import { useQueueStore } from "./store/queue";

export const generateUUID = () => {
  let
    d = new Date().getTime(),
    d2 = (performance && performance.now && (performance.now() * 1000)) || 0;
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    let r = Math.random() * 16;
    if (d > 0) {
      r = (d + r) % 16 | 0;
      d = Math.floor(d / 16);
    } else {
      r = (d2 + r) % 16 | 0;
      d2 = Math.floor(d2 / 16);
    }
    return (c == 'x' ? r : (r & 0x7 | 0x8)).toString(16);
  });
};

export const randomIntegerInRange = (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min;

export const downloadImage = (imageData: string, name: string) => {
  const a = document.createElement('a');
  a.download = `${name}`;
  a.href = imageData;
  a.click();
}

export const generateRandomSeed = () => randomIntegerInRange(0, 2147483640);

// return an string result of the amount of minutes and seconds from a duration in milliseconds
export const getDuration = (duration: number) => {
  const minutes = Math.floor(duration / 60000);
  const seconds = Math.floor((duration % 60000) / 1000);
  return `${minutes}m ${seconds}s`;
}

const DATE_UNITS = {
  day: 86400,
  hour: 3600,
  minute: 60,
  second: 1
}

const getSecondsDiff = (timestamp: number) => (Date.now() - timestamp) / 1000
const getUnitAndValueDate = (secondsElapsed: number) => {
  for (const [unit, secondsInUnit] of Object.entries(DATE_UNITS)) {
    if (secondsElapsed >= secondsInUnit || unit === "second") {
      const value = Math.floor(secondsElapsed / secondsInUnit) * -1
      return { value, unit }
    }
  }
}

export const getTimeAgo = (timestamp: number) => {
  const rtf = new Intl.RelativeTimeFormat('en', { numeric: 'auto', style: 'long' });

  const secondsElapsed = getSecondsDiff(timestamp)
  const unitAndValue = getUnitAndValueDate(secondsElapsed)
  if(!unitAndValue) return '';
  return rtf.format(unitAndValue.value, unitAndValue.unit as Intl.RelativeTimeFormatUnit)
}

export const playSound = (sound: SoundName) => {
  const audio = new Audio(SOUNDS_MAP[sound]);
  audio.volume = 0.5
  audio.play().catch(() => console.log('Interact with the page first to play sound'));
}

export const useIsServerOnline = () => {
  const isServerOnline = ref(false);
  const timeCheckedServer = ref(0);

  const queueStore = useQueueStore();
  const checkIsServer = async () => {
    if(queueStore.running) return;
    const previousVal = isServerOnline.value;
    isServerOnline.value = await checkServer()
    timeCheckedServer.value = Date.now();
    if(isServerOnline.value && isServerOnline.value !== previousVal) {
      playSound('server-online')
    }
  }
  onMounted(() => {
    checkIsServer();
  }),
  useIntervalFn(async () =>{
    await checkIsServer();
  }, 8000)

  return {
    isServerOnline,
    timeCheckedServer
  };
}