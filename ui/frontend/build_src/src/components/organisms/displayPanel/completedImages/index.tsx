import React from "react";

import { CompletedImagesType } from "../index";

import {
  completedImagesMain,
  completedImagesList,
  imageContain,
  RemoveButton,
  // @ts-expect-error
} from "./completedImages.css.ts";

interface CurrentDisplayProps {
  images: CompletedImagesType[] | null;
  setCurrentDisplay: (image: CompletedImagesType) => void;
  removeImages: () => void;
}

export default function CompletedImages({
  images,
  setCurrentDisplay,
  removeImages,
}: CurrentDisplayProps) {
  const _handleSetCurrentDisplay = (index: number) => {
    const image = images![index];
    setCurrentDisplay(image);
  };

  return (
    <div className={completedImagesMain}>
      {/* Adjust the dom do we dont do this check twice */}
      {images != null && images.length > 0 && (
        <button className={RemoveButton} onClick={
          () => {
            removeImages();
          }
        }>REMOVE</button>
      )}
      <ul className={completedImagesList}>
        {images != null &&
          images.map((image, index) => {
            if (void 0 === image) {
              console.warn(`image ${index} is undefined`);
              return null;
            }

            return (
              <li key={image.id}>
                <button
                  className={imageContain}
                  onClick={() => {
                    _handleSetCurrentDisplay(index);
                  }}
                >
                  <img src={image.data} alt={image.info.prompt} />
                </button>
              </li>
            );
          })}
      </ul>
    </div>

  );
}
