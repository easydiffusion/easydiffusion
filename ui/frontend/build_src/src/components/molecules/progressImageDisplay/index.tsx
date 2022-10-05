import React, { useState, useEffect } from "react";

import { API_URL } from "../../../api";
import { useProgressImages } from "../../../stores/progressImagesStore";
import { useImageDisplay } from "../../../stores/imageDisplayStore";

import {
  progressImageDisplayStyle
} from "./progressImageDisplay.css";

interface ProgressMainDisplayProps {
  batchId: string;
  seed?: string;
}

interface ProgressListDisplayProps {
  batchId: string;
  seed: string;
  orientation: 'horizontal' | 'vertical';
}

//batchId, seed,
function ProgressImageList({ batchId, seed, orientation }: ProgressListDisplayProps) {

  const getProgressImageList = useProgressImages((state) => state.getProgressImageList);
  const list = getProgressImageList(batchId, seed);

  const setCurrentImage = useImageDisplay((state) => state.setCurrentImage);
  const setProgressAsCurrent = (progressId: string) => {
    // console.log('setProgressAsCurrent - batchId', batchId);
    // console.log('progressId', progressId);
    if (batchId != null && seed != null) {
      setCurrentImage({ batchId, progressId, seed });
    }
  }
  return (
    <div className={progressImageDisplayStyle({ orientation })}>
      {
        list.map((image: any) => {
          // console.log('image.data', image.data);

          return <img
            key={image.id}
            // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
            src={`${API_URL}${image.data}`}
            alt={image.id}
            onClick={() => { setProgressAsCurrent(image.id) }} />;
        })
      }
    </div>
  );
}


export default function ProgressImageDisplay({ batchId, seed }: ProgressMainDisplayProps) {

  const { progressImageBySeed } = useProgressImages((state) => state.getProgressImageBatch(batchId)) ?? {};
  const [batchSeeds, setBatchSeeds] = useState<string[]>([]);

  useEffect(() => {
    if (progressImageBySeed != null) {

      const keys = Object.entries(progressImageBySeed).map(([key, value]) => {
        return key;
      });

      setBatchSeeds(keys);
    }

  }, [progressImageBySeed])

  return (
    <div>
      {void 0 != seed && (
        <ProgressImageList key={seed} batchId={batchId} seed={seed} orientation='vertical' />
      )}

      {void 0 == seed && (
        <div>
          {batchSeeds?.map((seed: any) => {
            return <ProgressImageList key={seed} batchId={batchId} seed={seed} orientation='horizontal' />
          })}
        </div>
      )}
    </div>
  );
};