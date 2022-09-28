
import React from "react";

import CurrentDisplay from "./currentDisplay";
import CompletedImages from "./completedImages";

import {
  displayPanel,
  displayContainer,
  previousImages,
  // @ts-expect-error
} from "./displayPanel.css.ts";


const idDelim = "_batch";

export default function DisplayPanel() {

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
    <div className={displayPanel}>
      <div className={displayContainer}>
        <CurrentDisplay
        ></CurrentDisplay>
      </div>

      <div className={previousImages}>
        <CompletedImages
        ></CompletedImages>
      </div>

    </div>
  );
}
