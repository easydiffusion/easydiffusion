import React from "react";
import GeneratedImage from "../../../molecules/generatedImage";
import {
  ImageRequest,
  useImageCreate,
} from "../../../../stores/imageCreateStore";

import { CompletedImagesType } from "../index";

type CurrentDisplayProps = {
  image: CompletedImagesType | null;
};

export default function CurrentDisplay({ image }: CurrentDisplayProps) {
  const { info, data } = image || { info: null, data: null };

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
    } = info!;

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
    link.href = data!;
    link.click();
  };

  const _handleUseAsInput = () => {
    setRequestOption("init_image", data);
  };

  return (
    <div className="current-display">
      {image && (
        <div>
          <p> {info!.prompt}</p>
          <GeneratedImage imageData={data!} metadata={info!}></GeneratedImage>

          <div>
            <button onClick={_handleSave}>Save</button>
            <button onClick={_handleUseAsInput}>Use as Input</button>
          </div>
        </div>
      )}
      <div></div>
    </div>
  );
}
