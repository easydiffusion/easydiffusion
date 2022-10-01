import React, { useState } from "react";


import { useImageDisplay } from "../../../stores/imageDisplayStore";

import {
  completedImagesMain,
  completedImagesContent,
  completedImagesList,
  imageContain,
} from "./completedImages.css";

import {
  buttonStyle
} from "../../_recipes/button.css";

import { Transition } from '@headlessui/react'


import {
  tabStyles
} from "../../_recipes/tabs_headless.css";

export default function CompletedImages(

) {

  const [isShowing, setIsShowing] = useState(false)


  const images = useImageDisplay((state) => state.images);
  const setCurrentImage = useImageDisplay((state) => state.setCurrentImage);
  const clearDisplay = useImageDisplay((state) => state.clearDisplay);

  const removeImagesAll = () => {
    clearDisplay();
  };

  return (
    <div className={completedImagesMain}>
      {/* <button
        className={tabStyles({})}
        onClick={() => setIsShowing((isShowing) => !isShowing)}>
        {isShowing ? "Hide History" : "Show History"}
      </button> */}
      {/* <Transition
        show={isShowing}
      > */}

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
                    setCurrentImage(image);
                  }}
                >
                  <img src={image.data} alt={image.info.prompt} />
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
