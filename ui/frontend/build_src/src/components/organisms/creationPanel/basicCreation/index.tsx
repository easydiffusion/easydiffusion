import React, { ChangeEvent } from "react";
import { useImageCreate } from "../../../../store/imageCreateStore";

import {
  CreationBasicMain,
  PromptDisplay, // @ts-ignore
} from "./basicCreation.css.ts";

import SeedImage from "./seedImage";
import ActiveTags from "./activeTags";
import MakeButton from "./makeButton";

export default function BasicCreation() {
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
        <p>Prompt </p>
        <textarea value={promptText} onChange={handlePromptChange}></textarea>
      </div>

      <SeedImage></SeedImage>

      <ActiveTags></ActiveTags>

      <MakeButton></MakeButton>
    </div>
  );
}
