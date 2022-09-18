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
    debugger;
    const image = images![index];
    setCurrentDisplay(image);
  };

  console.log("COMP{LETED IMAGES", images);
  return (
    <div className={completedImagesMain}>
      {images &&
        images.map((image, index) => {
          // if (void 0 !== image) {
          //   return null;
          // }

          return (
            // <div className={imageContain} key={index} value={index} onClick={() => {
            //   debugger;
            //   const image = images[index];
            //   _handleSetCurrentDisplay(image);
            // }}>
            //   <img src={image.data} alt={image.info.prompt} />
            // </div>

            <button
              key={index}
              className={imageContain}
              onClick={() => {
                console.log("CLICKED", index);
                debugger;
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
