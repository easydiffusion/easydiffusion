import React from "react";


import { useImageDisplay } from "../../../../stores/imageDisplayStore";

import {
  completedImagesMain,
  completedImagesList,
  imageContain,
  RemoveButton,
} from "./completedImages.css";



export default function CompletedImages(

) {


  const images = useImageDisplay((state) => state.images);
  const setCurrentImage = useImageDisplay((state) => state.setCurrentImage);
  const clearDisplay = useImageDisplay((state) => state.clearDisplay);


  const removeImagesAll = () => {
    clearDisplay();
  };

  // const idDelim = "_batch";
  //   if (completedQueries.length > 0) {
  //     // map the completedImagesto a new array
  //     // and then set the state
  //     const temp = completedQueries
  //       .map((query, index) => {
  //         if (void 0 !== query) {
  //           return query.output.map((data: ImageOutput, index: number) => {
  //             return {
  //               id: `${completedIds[index]}${idDelim}-${data.seed}-${index}`,
  //               data: data.data,
  //               info: { ...query.request, seed: data.seed },
  //             };
  //           });
  //         }
  //       })
  //       .flat()
  //       .reverse()
  //       .filter((item) => void 0 !== item) as CompletedImagesType[]; // remove undefined items

  return (
    <div className={completedImagesMain}>
      {/* Adjust the dom do we dont do this check twice */}
      {images != null && images.length > 0 && (
        <button
          className={RemoveButton}
          onClick={() => {
            removeImagesAll();
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
  );
}
