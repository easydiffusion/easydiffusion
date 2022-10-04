import React, { useState, useEffect } from "react";

import { API_URL } from "../../../api";
import { useProgressImages } from "../../../stores/progressImagesStore";
import { useImageDisplay } from "../../../stores/imageDisplayStore";

import {
  root
} from "./progressImageDisplay.css";

interface ProgressImageDisplayProps {
  batchId: string;
}

export default function ProgressImageDisplay({ batchId }: ProgressImageDisplayProps) {
  const progressImages = useProgressImages((state) => state.getProgressImages(batchId));

  const setCurrentImage = useImageDisplay((state) => state.setCurrentImage);

  const setProgressAsCurrent = (progressId: string) => {
    console.log('setProgressAsCurrent - batchId', batchId);
    console.log('progressId', progressId);
    setCurrentImage({ batchId, progressId });
  }

  if (progressImages.length === 0) {
    return null;
  }

  return (
    <div className={root}>
      {progressImages.map((image) => {
        return <img
          key={image.id}
          src={`${API_URL}${image.data}`}
          alt={image.id}
          onClick={() => { setProgressAsCurrent(image.id) }} />;
      })}
    </div>
  );
};