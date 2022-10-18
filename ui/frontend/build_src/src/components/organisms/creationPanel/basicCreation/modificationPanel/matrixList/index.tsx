/* eslint-disable @typescript-eslint/naming-convention */
import React, { useEffect, useRef } from "react";


import { usePromptMatrix } from "../../../../../../stores/promptMatrixStore";
import { useImageCreate } from "../../../../../../stores/imageCreateStore";

import PromptTag from "../../../../../molecules/promptTag";

import {
  buttonStyle
} from "../../../../../_recipes/button.css";

import {
  matrixListMain,
  matrixListItem,
  matrixListPrompt,
  matrixListTags,
} from "./matrixList.css";

interface Props {
  className?: string;
}

export default function MatrixList({ className }: Props) {

  const promptsList = usePromptMatrix((state) => state.promptsList);
  const positivePrompt = useImageCreate((state) => state.getValueForRequestKey("prompt"));
  const negativePrompt = useImageCreate((state) => state.getValueForRequestKey("negative_prompt"));

  return (
    <div className={matrixListMain}>
      {promptsList.map((prompt) => (
        <div className={matrixListItem} key={prompt.queueId}>
          <div className={matrixListPrompt}>
            <p>
              + {positivePrompt}
            </p>
            <p>
              - {negativePrompt}
            </p>
          </div>
          <div className={matrixListTags}>
            {prompt.options.map((option) => (
              <PromptTag key={option.id} queueId={prompt.queueId} id={option.id} name={option.name} type={option.type} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}