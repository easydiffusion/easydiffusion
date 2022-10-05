import React, { useState, useEffect } from "react";

import { API_URL } from "../../../api";
import { useProgressImages } from "../../../stores/progressImagesStore";
import { useImageDisplay } from "../../../stores/imageDisplayStore";

import {
  progressImageDisplayStyle
} from "./progressImageDisplay.css";

interface ProgressImageDisplayProps {
  batchId: string;
  seed?: string;
  orientation: 'horizontal' | 'vertical';
}

function ProgressImageList({ batchId, seed }: Partial<ProgressImageDisplayProps>) {

  const progressImages = useProgressImages((state) => state.getProgressRecordImages(batchId!, seed!));
  const setCurrentImage = useImageDisplay((state) => state.setCurrentImage);
  const setProgressAsCurrent = (progressId: string) => {
    console.log('setProgressAsCurrent - batchId', batchId);
    console.log('progressId', progressId);
    if (batchId != null && seed != null) {
      setCurrentImage({ batchId, progressId, seed });
    }
  }
  return (
    <>
      {
        progressImages.map((image) => {
          return <img
            key={image.id}
            src={`${API_URL}${image.data}`}
            alt={image.id}
            onClick={() => { setProgressAsCurrent(image.id) }} />;
        })
      }
    </>
  );
}


export default function ProgressImageDisplay({ batchId, seed, orientation }: ProgressImageDisplayProps) {
  // const progressImages = useProgressImages((state) => state.getProgressImages(batchId));

  // if (progressImages.length === 0) {
  //   return null;
  // }
  const [isSingle, setIsSingle] = useState(true);


  const getProgressRecord = useProgressImages((state) => state.getProgressRecord);

  const records = getProgressRecord(batchId);

  useEffect(() => {
    ///seed is optional
    const sing = seed != null
    console.log('sing', sing);
    setIsSingle(sing);
  }, [seed])

  // useEffect(() => {
  //   console.log('ProgressImageDisplay - batchId', batchId);
  //   console.log('ProgressImageDisplay - seed', seed);

  //   getProgressRecord(batchId)

  // }, [batchId, seed])

  return (
    <div className={progressImageDisplayStyle({
      orientation
    })}>

      {isSingle && <ProgressImageList batchId={batchId} seed={seed} />}

      {!isSingle && (
        // <ProgressImageList batchId={batchId} />
        records?.map((record) => {
          return <ProgressImageList key={record.seed} batchId={batchId} seed={record.seed} />
        })

      )}

    </div>
  );
};