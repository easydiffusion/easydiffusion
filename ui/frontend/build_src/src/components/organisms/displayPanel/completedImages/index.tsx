import React from "react";

import { CompletedImagesType } from "../index";

import {
  completedImagesMain,
  imageContain, // @ts-expect-error
} from "./completedImages.css.ts";

interface CurrentDisplayProps {
  images: CompletedImagesType[] | null;
  setCurrentDisplay: (image: CompletedImagesType) => void;
}

export default function CompletedImages({
  images,
  setCurrentDisplay,
}: CurrentDisplayProps) {
  const _handleSetCurrentDisplay = (index: number) => {
    const image = images![index];
    setCurrentDisplay(image);
  };

  return (
    <div className={completedImagesMain}>
      {images != null &&
        images.map((image, index) => {
          if (void 0 === image) {
            console.warn(`image ${index} is undefined`);
            return null;
          }

          return (
            <button
              key={index}
              className={imageContain}
              onClick={() => {
                _handleSetCurrentDisplay(index);
              }}
            >
              <img src={image.data} alt={image.info.prompt} />
            </button>
          );
        })}
    </div>
  );
}
