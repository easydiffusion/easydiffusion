import React, { useEffect, useState } from "react";

import { FetchingStates, useImageFetching } from "../../../stores/imageFetchingStore";
import { useImageDisplay } from "../../../stores/imageDisplayStore";

import { API_URL } from "../../../api";

import {
  currentDisplayMain,
} from './currentDisplay.css';

import ImageDisplay from "./imageDisplay";

const IdleDisplay = () => {
  return (
    <h4 className="no-image">Try Making a new image!</h4>
  );
};

const LoadingDisplay = ({ images }: { images: string[] }) => {

  return (
    <>
      {images.map((image, index) => {
        if (index == images.length - 1) {
          return (
            <img src={`${API_URL}${image}`} key={index} />
          )
        }
      })
      }
    </>
  );
};

export default function CurrentDisplay() {

  const status = useImageFetching((state) => state.status);
  const currentImage = useImageDisplay((state) => state.currentImage);

  const progressImages = useImageFetching((state) => state.progressImages);

  return (
    <div className={currentDisplayMain}>

      {(currentImage == null) && <IdleDisplay />}
      {/* {(status === FetchingStates.FETCHING || status === FetchingStates.PROGRESSING) && <LoadingDisplay />}
      {(currentImage != null) && <ImageDisplay info={currentImage?.info} data={currentImage?.data} />}  */}

      {
        (progressImages.length > 0)
          ? <LoadingDisplay images={progressImages} />
          : (currentImage != null) && <ImageDisplay info={currentImage?.info} data={currentImage?.data} />
      }

    </div>
  );
}