/* eslint-disable @typescript-eslint/no-non-null-assertion */
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

// const LoadingDisplay = ({ images }: { images: string[] }) => {

//   return (
//     <>
//       {images.map((image, index) => {
//         if (index == images.length - 1) {
//           return (
//             // TODO: make and 'ApiImage' component
//             <img src={`${API_URL}${image}`} key={index} />
//           )
//         }
//       })
//       }
//     </>
//   );
// };

export default function CurrentDisplay() {

  const status = useImageFetching((state) => state.status);
  const imageKeys = useImageDisplay((state) => state.currentImageKeys);

  console.log('imageKeys', imageKeys);


  return (
    <div className={currentDisplayMain}>

      {(imageKeys == null) && <IdleDisplay />}

      {(imageKeys != null) && <ImageDisplay batchId={imageKeys.batchId} imageId={imageKeys.imageId} progressId={imageKeys.progressId} />}

    </div>
  );
}