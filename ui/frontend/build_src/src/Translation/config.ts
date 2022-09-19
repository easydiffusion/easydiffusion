import i18n from "i18next";
// this should be updated to an interface
import translation from "./locales/en/home.json";
import { initReactI18next } from "react-i18next";

export const resources = {
  en: {
    translation,
  },
} as const;

i18n.use(initReactI18next).init({
  lng: "en",
  interpolation: {
    escapeValue: false,
  },
  resources,
});
