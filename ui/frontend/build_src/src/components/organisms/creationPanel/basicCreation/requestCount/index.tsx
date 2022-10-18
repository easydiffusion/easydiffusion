import React, { useEffect, useState } from "react";

import { useImageCreate } from "../../../../../stores/imageCreateStore";
import { usePromptMatrix } from "../../../../../stores/promptMatrixStore";

import {
  RequestCountMain
} from "./requestCount.css";

interface Props {
  className?: string;
}


export default function RequestCount({ className }: Props) {

  const promptsList = usePromptMatrix((state) => state.promptsList);
  const [totalRequests, setTotalRequests] = useState(1);
  const [totalImages, setTotalImages] = useState(1);
  const [imageText, setImageText] = useState("image");
  const [requestText, setRequestText] = useState("request");

  const parallelCount = useImageCreate((state) => state.parallelCount);
  const individualOutputs = useImageCreate((state) =>
    state.getValueForRequestKey("num_outputs")
  );

  useEffect(() => {
    let total = Math.ceil(individualOutputs / parallelCount);

    total *= (promptsList.length + 1);
    const totalOutputs = individualOutputs * (promptsList.length + 1);

    setTotalRequests(total);
    setTotalImages(totalOutputs)

    if (individualOutputs === totalOutputs) {
      setImageText("image");
    } else {
      setImageText("images");
    }

    if (total === 1) {
      setRequestText("request");
    } else {
      setRequestText("requests");
    }

  }, [setTotalRequests, setTotalImages, parallelCount, individualOutputs, promptsList]);

  return (
    <div className={[className, RequestCountMain].join(" ")}>
      <p>Making {totalImages} {imageText} in {totalRequests} {requestText}</p>
    </div>
  );
};