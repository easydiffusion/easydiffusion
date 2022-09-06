import { NConfigProvider, GlobalThemeOverrides, darkTheme } from 'naive-ui'


export type Prompt = {
  text: string
  modifiers: string[]
  finalPrompt: string
}

export type GeneratedImageData = {
  id: string;
  prompt: Prompt;
  seed: string;
  steps: string;
  guidance: string;
  width: string;
  height: string;
  createdAt: number;
  startedAt: number;
  queuedAt: number;
  elapsedMs: number;
  imageUrl?: string;
  imageFilename?: string;
  // img2img params ->
  init_image?: string;
  prompt_strength?: string;
  mask?: string;
};

type storedInPiniaParams =
  | 'seed'
  | 'steps'
  | 'guidance'
  | 'width'
  | 'height'

export type PayloadToQueueImage = Omit<GeneratedImageData, storedInPiniaParams>

export const DATE_FORMAT = 'DD/MM/YYYY - HH:mm:ss';

export const DEFAULT_SEED = '-1';
export const DEFAULT_GUIDANCE = 9;
export const DEFAULT_STEPS = 60;
export const DEFAULT_WIDTH = 512;
export const DEFAULT_HEIGHT = 512;
export const DEFAULT_PROMPT = 'old harbour';
export const DEFAULT_MODIFIERS = [
  "Tone mapped",
  "Shiny",
  "Intricate",
  "Cinematic lighting",
  "Highly detailed",
  "Digital painting",
  "Trending on artstation",
  "Concept art",
  "Smooth",
  "Sharp focus",
  "Illustration",
  "by Terry Moore and Greg Rutkowski and Alphonse Mucha"
]

export const API_URL = import.meta.env.DEV ? 'http://localhost:9000' : '';
export const API_GENERATE_IMAGE_URL = `${API_URL}/image` as const;
export const API_GENERATE_HEALTH = `${API_URL}/ping` as const;

export const SOUNDS_MAP = {
  'server-online': '/sound/hangover-sound.mp3',
  'done': '/sound/notification-pretty-good.mp3',
  'one-image': '/sound/message-tone-checked-off.mp3',
}
export type SoundName = keyof typeof SOUNDS_MAP;


export const sizes = ["64", "128", "256", "512"] as const;
export type Size = typeof sizes[number];


export const naiveUiThemeOverrides = {
  common: {
    primaryColor: "rgba(99, 148, 226, 1)",
    primaryColorHover: "#5590FFFF",
    primaryColorPressed: "#5A9CCEFF",
    primaryColorSuppl: "red"
  },
  Button: {
    colorPrimary: "#002d74",
    colorHoverPrimary: "#0b2752FF",
    colorFocusPrimary: "#041022",
    textColorPrimary: "#FFFFFFDB",
    textColorHoverPrimary: "#FFFFFFF7",
    textColorPressedPrimary: "#FFFFFFFF",
    textColorFocusPrimary: "#FFFFFFFF",
    textColorDisabledPrimary: "#FFFFFFFF"
  },
  Slider: {
    fillColor: "rgba(25, 69, 163, 1)",
    fillColorHover: "rgba(35, 98, 198, 1)",
    handleColor: "#BCD0FFFF"
  }
} as GlobalThemeOverrides
