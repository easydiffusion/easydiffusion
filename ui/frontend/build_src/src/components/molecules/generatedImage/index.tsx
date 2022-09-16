import React, { useCallback } from "react";

import { ImageRequest, useImageCreate } from "../../../stores/imageCreateStore";

import {
  generatedImage,
  imageContain,
  image,
  saveButton,
  useButton, //@ts-ignore
} from "./generatedImage.css.ts";

type GeneretaedImageProps = {
  imageData: string;
  metadata: ImageRequest;
  className?: string;
};

export default function GeneratedImage({
  imageData,
  metadata,
  className,
}: GeneretaedImageProps) {
  const setRequestOption = useImageCreate((state) => state.setRequestOptions);

  const createFileName = () => {
    const {
      prompt,
      seed,
      num_inference_steps,
      guidance_scale,
      use_face_correction,
      use_upscale,
      width,
      height,
    } = metadata;

    //Most important information is the prompt
    let underscoreName = prompt.replace(/[^a-zA-Z0-9]/g, "_");
    underscoreName = underscoreName.substring(0, 100);
    // name and the top level metadata
    let fileName = `${underscoreName}_Seed-${seed}_Steps-${num_inference_steps}_Guidance-${guidance_scale}`;
    // Add the face correction and upscale
    if (use_face_correction) {
      fileName += `_FaceCorrection-${use_face_correction}`;
    }
    if (use_upscale) {
      fileName += `_Upscale-${use_upscale}`;
    }
    // Add the width and height
    fileName += `_${width}x${height}`;
    // add the file extension
    fileName += `.png`;
    // return fileName
    return fileName;
  };

  const _handleSave = () => {
    const link = document.createElement("a");
    link.download = createFileName();
    link.href = imageData;
    link.click();
  };

  const _handleUseAsInput = () => {
    setRequestOption("init_image", imageData);
  };

  // className={[statusClass, className].join(" ")}

  return (
    <div className={[generatedImage, className].join(" ")}>
      <p>{metadata.prompt}</p>
      <div className={imageContain}>
        <img className={image} src={imageData} alt="generated" />
        <button className={saveButton} onClick={_handleSave}>
          Save
        </button>
        <button className={useButton} onClick={_handleUseAsInput}>
          Use as Input
        </button>
      </div>
    </div>
  );
}
