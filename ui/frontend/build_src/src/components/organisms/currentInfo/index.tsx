/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/naming-convention */
/* eslint-disable @typescript-eslint/consistent-type-assertions */
import React, { useEffect, useState } from "react";

import { useImageDisplay } from "../../../stores/imageDisplayStore";
import { useProgressImages } from "../../../stores/progressImagesStore";
import { useCreatedMedia } from "../../../stores/createdMediaStore";
import { currentInfoMain } from "./currentInfo.css";

export default function CurrentInfo() {

  const [batchId, setBatchId] = useState<string>('');
  const [imageId, setImageId] = useState<string>('');

  const [gScale, setGScale] = useState<number>(1);
  const [prompt, setPrompt] = useState<string>('');
  const [negPrompt, setNegPrompt] = useState<string>('');
  const [seed, setSeed] = useState<number>(0);

  const imageKeys = useImageDisplay((state) => state.currentImageKeys);

  const progressImages = useProgressImages((state) => state.getProgressImages(batchId));
  const createdMedia = useCreatedMedia((state) => state.getCreatedMedia(batchId));


  useEffect(() => {
    if (imageKeys != null) {
      setBatchId(imageKeys.batchId);
      setImageId(imageKeys.imageId);
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

          {progressImages?.map((image, index) => {

            return (
              <div key={image.id}>
                <img src={image.data} />
              </div>
            );
          })

          }

        </>
      )}
    </div>);
}