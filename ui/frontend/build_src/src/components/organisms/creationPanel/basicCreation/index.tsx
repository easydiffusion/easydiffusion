import React, { ChangeEvent } from "react";
import { useImageCreate } from "../../../../stores/imageCreateStore";

import {
  CreationBasicMain,
  PromptDisplay,
} from "./basicCreation.css";

import MakeButton from "../../../molecules/makeButton";

import PromptCreator from "./promptCreator";
// import CreationActions from "./creationActions";
import SeedImage from "./seedImage";
import ActiveTags from "./promptCreator/activeTags";

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
      <MakeButton></MakeButton>
      <PromptCreator></PromptCreator>
      <SeedImage></SeedImage>
    </div>
  );
}
