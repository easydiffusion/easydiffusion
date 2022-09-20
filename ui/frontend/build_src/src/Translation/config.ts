import i18n from "i18next";
// this should be updated to an interface
import ENTranslation from "./locales/en/home.json";
import ESTranslation from "./locales/es/home.json";
import { initReactI18next } from "react-i18next";

export const resources = {
  en: {
    translation: ENTranslation,
  },
  es: {
    translation: ESTranslation,
  },
} as const;
i18n.use(initReactI18next).init({
  lng: "en",
  interpolation: {
    escapeValue: false,
  },
  resources,
}).then(() => {
  console.log("i18n initialized");
}).catch((err) => {
  console.error("i18n initialization failed", err);
}).finally(() => {
  console.log("i18n initialization finished");
});