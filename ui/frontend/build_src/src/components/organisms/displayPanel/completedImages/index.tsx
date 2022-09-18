import React from "react";

import { CompletedImagesType } from "../index";

type CurrentDisplayProps = {
  images: CompletedImagesType[] | null;
  setCurrentDisplay: (image: CompletedImagesType) => void;
};

import {
  completedImagesMain,
  imageContain, //@ts-ignore
} from "./completedImages.css.ts";

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
      {images &&
        images.map((image, index) => {
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
