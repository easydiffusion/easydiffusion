/* eslint-disable multiline-ternary */
/* eslint-disable @typescript-eslint/naming-convention */
import React from "react";
import GeneratedImage from "../../../molecules/generatedImage";
import {
  ImageRequest,
  useImageCreate,
} from "../../../../stores/imageCreateStore";

import { CompletedImagesType } from "../index";

interface CurrentDisplayProps {
  isLoading: boolean;
  image: CompletedImagesType | null;
}

export default function CurrentDisplay({
  isLoading,
  image,
}: CurrentDisplayProps) {
  const { info, data } = image ?? {};

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

    // Most important information is the prompt
    let underscoreName = prompt.replace(/[^a-zA-Z0-9]/g, "_");
    underscoreName = underscoreName.substring(0, 100);
    // name and the top level metadata
    let fileName = `${underscoreName}_Seed-${seed}_Steps-${num_inference_steps}_Guidance-${guidance_scale}`;
    // Add the face correction and upscale
    if (typeof use_face_correction == "string") {
      fileName += `_FaceCorrection-${use_face_correction}`;
    }
    if (typeof use_upscale == "string") {
      fileName += `_Upscale-${use_upscale}`;
    }
    // Add the width and height
    fileName += `_${width}x${height}`;
    // add the file extension
    fileName += ".png";
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
      {isLoading ? (
        <h4 className="loading">Loading...</h4>
      ) : (
        (image !== null && (
          // eslint-disable-next-line @typescript-eslint/strict-boolean-expressions
          <div>
            <p> {info?.prompt}</p>
            <GeneratedImage imageData={data} metadata={info}></GeneratedImage>
            <div>
              <button onClick={_handleSave}>Save</button>
              <button onClick={_handleUseAsInput}>Use as Input</button>
            </div>
          </div>
        )) || <h4 className="no-image">Try Making a new image!</h4>
      )}
    </div>
  );
}
