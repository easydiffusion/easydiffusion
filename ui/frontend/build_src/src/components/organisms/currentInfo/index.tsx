/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/naming-convention */
/* eslint-disable @typescript-eslint/consistent-type-assertions */
import React, { useEffect, useState } from "react";

import { useImageDisplay } from "../../../stores/imageDisplayStore";
import { useCreatedMedia } from "../../../stores/createdMediaStore";
import { currentInfoMain } from "./currentInfo.css";

import ProgressImageDisplay from '../../molecules/progressImageDisplay';
// function CurrentActions() {
//   const createFileName = () => {
//     const {
//       prompt,
//       negative_prompt,
//       seed,
//       num_inference_steps,
//       guidance_scale,
//       use_face_correction,
//       use_upscale,
//       width,
//       height,
//     } = info;

//     // Most important information is the prompt
//     let underscoreName = prompt.replace(/[^a-zA-Z0-9]/g, "_");
//     underscoreName = underscoreName.substring(0, 100);
//     // name and the top level metadata
//     let fileName = `${underscoreName}_Seed-${seed}_Steps-${num_inference_steps}_Guidance-${guidance_scale}`;
//     // Add the face correction and upscale
//     if (typeof use_face_correction == "string") {
//       fileName += `_FaceCorrection-${use_face_correction}`;
//     }
//     if (typeof use_upscale == "string") {
//       fileName += `_Upscale-${use_upscale}`;
//     }
//     // Add the width and height
//     fileName += `_${width}x${height}`;
//     // add the file extension
//     fileName += ".png";
//     // return fileName
//     return fileName;
//   };

//   const setRequestOption = useImageCreate((state) => state.setRequestOptions);

//   const _handleSave = () => {
//     const link = document.createElement("a");
//     link.download = createFileName();
//     link.href = data ?? "";
//     link.click();
//   };

//   const _handleUseAsInput = () => {
//     setRequestOption("init_image", data);
//   };


//   return (

//   )
// }


export default function CurrentInfo() {

  const [batchId, setBatchId] = useState<string>('');
  const [imageId, setImageId] = useState<string>('');

  const [gScale, setGScale] = useState<number>(1);
  const [prompt, setPrompt] = useState<string>('');
  const [negPrompt, setNegPrompt] = useState<string>('');
  const [seed, setSeed] = useState<number>(0);

  const imageKeys = useImageDisplay((state) => state.currentImageKeys);

  // const progressImages = useProgressImages((state) => state.getProgressImages(batchId));
  const createdMedia = useCreatedMedia((state) => state.getCreatedMedia(batchId));

  useEffect(() => {
    if (imageKeys != null) {
      setBatchId(imageKeys.batchId);
    }
  }, [imageKeys]);

  useEffect(() => {
    if (createdMedia != null) {
      const {
        info: {
          guidance_scale,
          prompt,
          negative_prompt,
          seed
        }
      } = createdMedia;
      setGScale(guidance_scale);
      setPrompt(prompt);
      setNegPrompt(negative_prompt);
      setSeed(seed);
    }
  }, [createdMedia]);



  return (
    <div className={currentInfoMain}>
      {batchId != '' && (
        <>
          <div>id: {batchId}</div>
          <div>guidance_scale: {gScale}</div>
          <div>prompt: {prompt}</div>
          <div>negative_prompt: {negPrompt}</div>
          <div>seed: {seed}</div>

          <ProgressImageDisplay batchId={batchId}></ProgressImageDisplay>

        </>
      )}
    </div>);
}