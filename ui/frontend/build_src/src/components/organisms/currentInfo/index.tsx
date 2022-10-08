/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/naming-convention */
/* eslint-disable @typescript-eslint/consistent-type-assertions */
import React, { useEffect, useState } from "react";

import { useImageDisplay } from "../../../stores/imageDisplayStore";
import { useCreatedMedia } from "../../../stores/createdMediaStore";
import { currentInfoMain } from "./currentInfo.css";

import ProgressImageDisplay from '../../molecules/progressImageDisplay';


export default function CurrentInfo() {

  const [batchId, setBatchId] = useState<string>('');
  const [imageId, setImageId] = useState<string>('');

  const [gScale, setGScale] = useState<number>(1);
  const [prompt, setPrompt] = useState<string>('');
  const [negPrompt, setNegPrompt] = useState<string>('');
  const [seed, setSeed] = useState<number>(-1);

  const imageKeys = useImageDisplay((state) => state.currentImageKeys);

  // const progressImages = useProgressImages((state) => state.getProgressImages(batchId));
  const createdMedia = useCreatedMedia((state) => state.getCreatedMedia(batchId, seed));

  useEffect(() => {
    if (imageKeys != null) {
      console.log('imageKeys', imageKeys);
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
      console.log('SET FOR NEW MEDIA', seed)
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

          <ProgressImageDisplay batchId={batchId} seed={seed}></ProgressImageDisplay>

        </>
      )}
    </div>);
}