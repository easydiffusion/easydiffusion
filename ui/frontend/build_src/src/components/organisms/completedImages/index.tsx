import React, { useState, useEffect } from "react";
import { useImageDisplay } from "../../../stores/imageDisplayStore";

import { imageDataObject, useCreatedMedia } from "../../../stores/createdMediaStore";

import {
  completedImagesMain,
  completedImagesContent,
  completedImagesList,
  imageContain,
} from "./completedImages.css";

import {
  buttonStyle
} from "../../_recipes/button.css";
import { ImageRequest } from "../../../api/api.d";


interface completedImageObject extends imageDataObject {
  batchId: string;
  info: ImageRequest;
}

export default function CompletedImages() {

  const [images, setImages] = useState<completedImageObject[]>([])

  const setCurrentImage = useImageDisplay((state) => state.setCurrentImage);
  const clearDisplay = useImageDisplay((state) => state.clearDisplay);

  const createdMediaList = useCreatedMedia((state) => state.createdMediaList);

  useEffect(() => {
    const tempImages: any = [];
    createdMediaList?.forEach((media) => {
      const { data } = media;
      data?.forEach(element => {
        tempImages.push({ batchId: media.batchId, id: element.id, data: element.data, info: media.info })
      });
    })

    setImages(tempImages);
  }, [createdMediaList])

  const removeImagesAll = () => {
    clearDisplay();
  };

  return (
    <div className={completedImagesMain}>

      <div className={completedImagesContent}>
        {/* Adjust the dom do we dont do this check twice */}
        {images != null && images.length > 0 && (
          <button
            className={buttonStyle()}
            onClick={() => {
              removeImagesAll();
            }}
          >
            REMOVE ALL
          </button>
        )}
        <ul className={completedImagesList}>
          {images?.map((image, index) => {
            if (void 0 === image) {
              console.warn(`image ${index} is undefined`);
              return null;
            }


            return (
              <li key={image.id}>
                <button
                  className={imageContain}
                  onClick={() => {
                    setCurrentImage({ batchId: image.batchId, imageId: image.id, seed: image.info.seed });
                  }}
                >
                  <img src={image.data} />
                </button>
              </li>
            );
          })}
        </ul>
      </div>
      {/* </Transition> */}
    </div>
  );
}
