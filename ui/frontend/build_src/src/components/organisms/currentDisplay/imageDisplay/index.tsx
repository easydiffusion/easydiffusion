/* eslint-disable @typescript-eslint/naming-convention */
import React, { useEffect } from "react";

import {
  API_URL
} from "../../../../api";


import {
  imageDataObject,
  useCreatedMedia
} from "../../../../stores/createdMediaStore";

import {
  useImageCreate
} from "../../../../stores/imageCreateStore";
import { useProgressImages } from "../../../../stores/progressImagesStore";
import { CompletedImageIds } from "../../../../stores/imageDisplayStore";

import GeneratedImage from "../../../molecules/generatedImage";

import {
  imageDisplayMain,
  imageDisplayContainer,
  imageDisplayCenter,
  imageDisplayContent,
} from './imageDisplay.css';

import {
  buttonStyle
} from "../../../_recipes/button.css";
import { ImageRequest } from "../../../../api/api.d";

interface ImageActionsProps {
  info: ImageRequest;
  data: string;

}

function ImageActions({ info, data }: ImageActionsProps) {
  const createFileName = () => {
    const {
      prompt,
      negative_prompt,
      seed,
      num_inference_steps,
      guidance_scale,
      use_face_correction,
      use_upscale,
      width,
      height,
    } = info;

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

  const setRequestOption = useImageCreate((state) => state.setRequestOptions);

  const _handleSave = () => {
    const link = document.createElement("a");
    link.download = createFileName();
    link.href = data ?? "";
    link.click();
  };

  const _handleUseAsInput = () => {
    setRequestOption("init_image", data);
  };

  return (
    <div>
      <button className={buttonStyle(

      )} onClick={_handleSave}>Save</button>
      <button className={buttonStyle(
        {
          color: "secondary",
          type: "outline",
        }
      )} onClick={_handleUseAsInput}>Use as Input</button>
    </div>
  )
}


export default function ImageDisplay({ batchId, imageId, progressId }: CompletedImageIds) {


  const getCreatedMedia = useCreatedMedia((state) => state.getCreatedMedia);
  const getProgressImages = useProgressImages((state) => state.getProgressImages);

  // eslint-disable-next-line @typescript-eslint/consistent-type-assertions
  const [curMedia, setCurMedia] = React.useState<imageDataObject | null>(null);
  const [info, setInfo] = React.useState<ImageRequest | null>(null);

  useEffect(() => {
    if (batchId !== null) {
      // we always want the created media so we can get the info
      const created = getCreatedMedia(batchId);
      setInfo(created?.info ?? null);

      if (imageId !== null) {
        const curImage = created?.data?.filter((media) => media.id == imageId)[0] ?? null;
        setCurMedia(curImage);
      }

      if (void 0 != progressId) {
        const curImage = getProgressImages(batchId)?.filter((media) => media.id == progressId)[0] ?? null;
        console.log('getProgressImages curImage', curImage);
        setCurMedia({
          ...curImage,
          data: `${API_URL}${curImage.data}`
        });
      }
    }
  }, [batchId, imageId, progressId, getCreatedMedia, getProgressImages]);

  if (curMedia == null) {
    return null;
  }

  return (
    <div className={imageDisplayMain}>
      <div className={imageDisplayContainer}>
        <div className={imageDisplayCenter}>
          <div className={imageDisplayContent}>
            <GeneratedImage
              imageData={curMedia.data}
            ></GeneratedImage>

            {(info != null) && <ImageActions info={info} data={curMedia.data}></ImageActions>}

          </div>
        </div>
      </div>
    </div>
  );
}