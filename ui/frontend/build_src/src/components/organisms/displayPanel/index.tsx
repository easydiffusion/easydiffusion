import React, { useEffect, useState, useRef } from "react";
import { useImageQueue } from "../../../stores/imageQueueStore";

import { ImageRequest, useImageCreate } from "../../../stores/imageCreateStore";

import { useQuery, useQueryClient } from "@tanstack/react-query";

import {
  doMakeImage,
  MakeImageKey,
  ImageReturnType,
  ImageOutput,
} from "../../../api";

import AudioDing from "./audioDing";

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

export interface CompletedImagesType {
  id: string;
  data: string;
  info: ImageRequest;
}

const idDelim = "_batch";

export default function DisplayPanel() {
  const dingRef = useRef<HTMLAudioElement>(null);
  const isSoundEnabled = useImageCreate((state) => state.isSoundEnabled());

  // @ts-expect-error
  const { id, options } = useImageQueue((state) => state.firstInQueue());
  const removeFirstInQueue = useImageQueue((state) => state.removeFirstInQueue);

  const [currentImage, setCurrentImage] = useState<CompletedImagesType | null>(
    null
  );

  const [isEnabled, setIsEnabled] = useState(false);

  const [isLoading, setIsLoading] = useState(true);

  const { status, data } = useQuery(
    [MakeImageKey, id],
    async () => await doMakeImage(options),
    {
      enabled: isEnabled,
    }
  );

  // update the enabled state when the id changes
  useEffect(() => {
    setIsEnabled(void 0 !== id);
  }, [id]);

  // helper for the loading state to be enabled aware
  useEffect(() => {
    if (isEnabled && status === "loading") {
      setIsLoading(true);
    } else {
      setIsLoading(false);
    }
  }, [isEnabled, status]);

  // this is where there loading actually happens
  useEffect(() => {
    // query is done
    if (status === "success") {
      // check to make sure that the image was created
      if (data.status === "succeeded") {
        if (isSoundEnabled) {
          // not awaiting the promise or error handling
          void dingRef.current?.play();
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
  const clearCachedIds = useImageQueue((state) => state.clearCachedIds);

  // this is where we generate the list of completed images
  useEffect(() => {
    const completedQueries = completedIds.map((id) => {
      const imageData = queryClient.getQueryData([MakeImageKey, id]);
      return imageData;
    }) as ImageReturnType[];

    if (completedQueries.length > 0) {
      // map the completedImagesto a new array
      // and then set the state
      const temp = completedQueries
        .map((query, index) => {
          if (void 0 !== query) {
            return query.output.map((data: ImageOutput, index: number) => {
              return {
                id: `${completedIds[index]}${idDelim}-${data.seed}-${index}`,
                data: data.data,
                info: { ...query.request, seed: data.seed },
              };
            });
          }
        })
        .flat()
        .reverse()
        .filter((item) => void 0 !== item) as CompletedImagesType[]; // remove undefined items

      setCompletedImages(temp);

      // could move this to the useEffect for completedImages
      if (temp.length > 0) {
        setCurrentImage(temp[0]);
      } else {
        setCurrentImage(null);
      }
    } else {
      setCompletedImages([]);
      setCurrentImage(null);
    }
  }, [setCompletedImages, setCurrentImage, queryClient, completedIds]);

  // this is how we remove them
  const removeImages = () => {
    completedIds.forEach((id) => {
      queryClient.removeQueries([MakeImageKey, id]);
    });
    clearCachedIds();
  };

  return (
    <div className={displayPanel}>
      <AudioDing ref={dingRef}></AudioDing>
      <div className={displayContainer}>
        <CurrentDisplay
          isLoading={isLoading}
          image={currentImage}
        ></CurrentDisplay>
      </div>
      <div className={previousImages}>
        <CompletedImages
          removeImages={removeImages}
          images={completedImages}
          setCurrentDisplay={setCurrentImage}
        ></CompletedImages>
      </div>
    </div>
  );
}
