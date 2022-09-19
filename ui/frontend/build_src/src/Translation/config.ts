import i18n from "i18next";
// this should be updated to an interface
import ENTranslation from "./locales/en/home.json";
import ESTranlation from "./locales/es/home.json";
import { initReactI18next } from "react-i18next";

export const resources = {
  en: {
    translation: ENTranslation,
  },
  es: {
    translation: ESTranlation,
  },
} as const;

i18n.use(initReactI18next).init({
  lng: "en",
  interpolation: {
    escapeValue: false,
  },
  resources,
});
