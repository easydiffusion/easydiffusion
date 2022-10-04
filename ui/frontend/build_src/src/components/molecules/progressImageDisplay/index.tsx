import React, { useState, useEffect } from "react";

import { API_URL } from "../../../api";
import { imageDataObject, useProgressImages } from "../../../stores/progressImagesStore";

import {
  root
} from "./progressImageDisplay.css";

interface ProgressImageDisplayProps {
  batchId: string;
}
export default function ProgressImageDisplay({ batchId }: ProgressImageDisplayProps) {
  const progressImages = useProgressImages((state) => state.getProgressImages(batchId));
  if (progressImages.length === 0) {
    return null;
  }

  console.log('progressImages redraw');

  return (
    <div className={root}>
      {progressImages.map((image) => {
        console.log('id', image.id);
        // TODO: make and 'ApiImage' component
        return <img key={image.id} src={`${API_URL}${image.data}`} alt={image.id} />;
      })}
    </div>
  );
};