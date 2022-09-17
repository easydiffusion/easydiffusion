import React, { useEffect, useState, useRef } from "react";
import { useImageQueue } from "../../../stores/imageQueueStore";

import { ImageRequest, useImageCreate } from "../../../stores/imageCreateStore";

import { useQuery, useQueryClient } from "@tanstack/react-query";

import { doMakeImage, MakeImageKey } from "../../../api";

import AudioDing from "./audioDing";

import GeneratedImage from "../../molecules/generatedImage";
// import DrawImage from "../../molecules/drawImage";

import {
  displayPanel,
  displayContainer,
  CurrentDisplay,
  previousImages,
  previousImage, //@ts-ignore
} from "./displayPanel.css.ts";

type CompletedImagesType = {
  id: string;
  data: string;
  info: ImageRequest;
};

export default function DisplayPanel() {
  const dingRef = useRef<HTMLAudioElement>(null);
  const isSoundEnabled = useImageCreate((state) => state.isSoundEnabled());

  const isInPaintingMode = useImageCreate((state) => state.isInpainting);

  /* FETCHING  */
  // @ts-ignore
  const { id, options } = useImageQueue((state) => state.firstInQueue());
  const removeFirstInQueue = useImageQueue((state) => state.removeFirstInQueue);
  const { status, data } = useQuery(
    [MakeImageKey, id],
    () => doMakeImage(options),
    {
      enabled: void 0 !== id,
    }
  );

  useEffect(() => {
    // query is done
    if (status === "success") {
      // check to make sure that the image was created
      if (data.status === "succeeded") {
        if (isSoundEnabled) {
          dingRef.current?.play();
        }
        removeFirstInQueue();
      }
    }
  }, [status, data, removeFirstInQueue, dingRef, isSoundEnabled]);

  /* COMPLETED IMAGES */

  const queryClient = useQueryClient();
  const [completedImages, setCompletedImages] = useState<CompletedImagesType[]>(
    []
  );
  const completedIds = useImageQueue((state) => state.completedImageIds);

  const init_image = useImageCreate((state) =>
    state.getValueForRequestKey("init_image")
  );

  useEffect(() => {
    const testReq = {} as ImageRequest;
    const completedQueries = completedIds.map((id) => {
      const imageData = queryClient.getQueryData([MakeImageKey, id]);
      return imageData;
    });

    if (completedQueries.length > 0) {
      // map the completedImagesto a new array
      // and then set the state
      const temp = completedQueries
        .map((query, index) => {
          if (void 0 !== query) {
            //@ts-ignore
            return query.output.map((data) => {
              // @ts-ignore
              return {
                id: `${completedIds[index]}-${data.seed}`,
                data: data.data,
                //@ts-ignore
                info: { ...query.request, seed: data.seed },
              };
            });
          }
        })
        .flat()
        .reverse();
      setCompletedImages(temp);
    } else {
      setCompletedImages([]);
    }
  }, [setCompletedImages, queryClient, completedIds]);

  return (
    <div className={displayPanel}>
      <AudioDing ref={dingRef}></AudioDing>
      <div className={displayContainer}>
        {/* {isInPaintingMode && <DrawImage imageData={init_image}></DrawImage>} */}

        {completedImages.length > 0 && (
          <>
            <div className={CurrentDisplay}>
              <GeneratedImage
                key={completedImages[0].id}
                imageData={completedImages[0].data}
                metadata={completedImages[0].info}
              />
            </div>

            <div className={previousImages}>
              {completedImages.map((image, index) => {
                if (void 0 !== image) {
                  if (index == 0) {
                    return null;
                  }

                  return (
                    <GeneratedImage
                      className={previousImage}
                      key={image.id}
                      imageData={image.data}
                      metadata={image.info}
                    />
                  );
                } else {
                  console.warn("image is undefined", image, index);
                  return null;
                }
              })}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
