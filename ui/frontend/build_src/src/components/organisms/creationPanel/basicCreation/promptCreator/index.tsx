import React from "react";

import { useImageCreate } from "../../../../../stores/imageCreateStore";

import {
  PromptCreatorMain,
} from "./promptCreator.css";


import { useTranslation } from "react-i18next";


export default function PromptCreator() {

  const { t } = useTranslation();

  const prompt = useImageCreate((state) => state.getValueForRequestKey("prompt"));
  const negativePrompt = useImageCreate((state) => state.getValueForRequestKey("negative_prompt"));
  const setRequestOptions = useImageCreate((state) => state.setRequestOptions);

  return (
    <div className={PromptCreatorMain}>
      <div>
        <p>{t("home.editor-title")}</p>
        <textarea value={prompt} onChange={(event) => {
          setRequestOptions('prompt', event.target.value);
        }}></textarea>
      </div>

      <div>
        <p>negative prompt</p>
        <input value={negativePrompt} onChange={(event) => {
          setRequestOptions('negative_prompt', event.target.value);
        }}></input>
      </div>

    </div >
  );
}