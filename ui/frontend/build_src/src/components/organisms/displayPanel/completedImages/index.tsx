import React from "react";

import { CompletedImagesType } from "../currentDisplay";


import { useImageDisplay } from "../../../../stores/imageDisplayStore";

import {
  completedImagesMain,
  completedImagesList,
  imageContain,
  RemoveButton,
  // @ts-expect-error
} from "./completedImages.css.ts";

// interface CurrentDisplayProps {
//   images: CompletedImagesType[] | null;
//   setCurrentDisplay: (image: CompletedImagesType) => void;
//   removeImages: () => void;
// }

export default function CompletedImages(
  //   {
  //   images,
  //   setCurrentDisplay,
  //   removeImages,
  // }: CurrentDisplayProps

) {


  const images = useImageDisplay((state) => state.images);

  // useEffect(() => {
  //   if (images.length > 0) {
  //     console.log("cur", images[0]);
  //     setCurrentImage(images[0]);
  //   } else {
  //     setCurrentImage(null);
  //   }
  // }, [images]);



  // const _handleSetCurrentDisplay = (index: number) => {
  //   const image = images![index];
  //   setCurrentDisplay(image);
  // };

  const removeImages = () => {
  };

  return (
    <div className={completedImagesMain}>
      {/* Adjust the dom do we dont do this check twice */}
      {images != null && images.length > 0 && (
        <button
          className={RemoveButton}
          onClick={() => {
            removeImages();
          }}
        >
          REMOVE
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
              // onClick={() => {
              //   _handleSetCurrentDisplay(index);
              // }}
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
