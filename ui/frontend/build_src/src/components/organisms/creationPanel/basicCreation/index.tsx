import React, { ChangeEvent } from "react";
import { useImageCreate } from "../../../../stores/imageCreateStore";

import {
  CreationBasicMain,
  PromptDisplay, // @ts-expect-error
} from "./basicCreation.css.ts";

import SeedImage from "./seedImage";
import ActiveTags from "./activeTags";
import MakeButton from "./makeButton";

import { useTranslation } from "react-i18next";

export default function BasicCreation() {
  const { t } = useTranslation();

  const promptText = useImageCreate((state) =>
    state.getValueForRequestKey("prompt")
  );
  const setRequestOption = useImageCreate((state) => state.setRequestOptions);

  const handlePromptChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    setRequestOption("prompt", event.target.value);
  };

  return (
    <div className={CreationBasicMain}>
      <div className={PromptDisplay}>
        <p>{t("home.editor-title")}</p>
        <textarea value={promptText} onChange={handlePromptChange}></textarea>
      </div>
      <MakeButton></MakeButton>

      <SeedImage></SeedImage>

      <ActiveTags></ActiveTags>
    </div>
  );
}
