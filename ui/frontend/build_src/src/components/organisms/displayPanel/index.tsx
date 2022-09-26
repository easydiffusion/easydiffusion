/* eslint-disable @typescript-eslint/strict-boolean-expressions */
import React, { useEffect, useState, useRef } from "react";
// import { useImageQueue } from "../../../stores/imageQueueStore";

// import { useImageFetching } from "../../../stores/imageFetchingStore";
// import { useImageDisplay } from "../../../stores/imageDisplayStore";
import { useImageDisplay } from "../../../stores/imageDisplayStore";

// import { ImageRequest, useImageCreate } from "../../../stores/imageCreateStore";

// import { useQuery, useQueryClient } from "@tanstack/react-query";


// import {
//   API_URL,
//   doMakeImage,
//   MakeImageKey,
//   ImageReturnType,
//   ImageOutput,
// } from "../../../api";

// import AudioDing from "../creationPanel/basicCreation/makeButton/audioDing";

// import GeneratedImage from "../../molecules/generatedImage";
// import DrawImage from "../../molecules/drawImage";

import CurrentDisplay from "./currentDisplay";
import CompletedImages from "./completedImages";

import {
  displayPanel,
  displayContainer,
  previousImages,
  // @ts-expect-error
} from "./displayPanel.css.ts";

// export interface CompletedImagesType {
//   id: string;
//   data: string;
//   info: ImageRequest;
// }

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
      {/* <AudioDing ref={dingRef}></AudioDing> */}
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
