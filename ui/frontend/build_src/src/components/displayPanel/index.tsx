import React, { useEffect, useState, useRef } from "react";
import { useImageQueue } from "../../store/imageQueueStore";

import { ImageRequest, useImageCreate } from "../../store/imageCreateStore";

import { useQuery, useQueryClient } from "@tanstack/react-query";

import { doMakeImage, MakeImageKey } from "../../api";

import AudioDing from "./audioDing";

import GeneratedImage from "./generatedImage";

import './displayPanel.css';

type CompletedImagesType = {
  id: string;
  data: string;
  info: ImageRequest;
};
export default function DisplayPanel() {
  const dingRef = useRef<HTMLAudioElement>(null);
  const isSoundEnabled = useImageCreate((state) => state.isSoundEnabled());

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
      debugger;
    } else {
      setCompletedImages([]);
    }
  }, [setCompletedImages, queryClient, completedIds]);

  return (
    <div className="display-panel">
      <h1>Display Panel</h1>
      <AudioDing ref={dingRef}></AudioDing>
      {completedImages.length > 0 && (
        <div id="display-container">

          <GeneratedImage
            key={completedImages[0].id}
            imageData={completedImages[0].data}
            metadata={completedImages[0].info}
          />

          <div id="previous-images">
            {completedImages.map((image, index) => {

              if (void 0 !== image) {
                if(index == 0){
                    return null;
                  }

                return (
                  <GeneratedImage
                    className="previous-image"
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

        </div>
      )}
    </div>
  );
}
