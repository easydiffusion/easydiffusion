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
  // const dingRef = useRef<HTMLAudioElement>(null);
  // const isSoundEnabled = useImageCreate((state) => state.isSoundEnabled());

  // // @ts-expect-error
  // const { id, options } = useImageQueue((state) => state.firstInQueue());
  // const removeFirstInQueue = useImageQueue((state) => state.removeFirstInQueue);



  // const [isEnabled, setIsEnabled] = useState(false);

  // const [isLoading, setIsLoading] = useState(true);

  // const { status, data } = useQuery(
  //   [MakeImageKey, id],
  //   async () => await doMakeImage(options),
  //   {
  //     enabled: isEnabled,
  //   }
  // );

  // const { status, data } = useEventSourceQuery(
  //   MakeImageKey,

  //   // [MakeImageKey, id],
  //   // async () => await doMakeImage(options),
  //   // {
  //   //   enabled: isEnabled,
  //   // }
  // );

  // update the enabled state when the id changes
  // useEffect(() => {
  //   setIsEnabled(void 0 !== id);
  // }, [id]);

  // const _handleStreamData = async (res: typeof ReadableStream) => {
  //   console.log("_handleStreamData");
  //   let reader;
  //   // @ts-expect-error
  //   if (res.body.locked) {
  //     console.log("locked");
  //   }
  //   else {
  //     reader = res.body.getReader();
  //   }

  //   console.log("reader", reader);

  //   const decoder = new TextDecoder();
  //   while (true) {
  //     const { done, value } = await reader.read();

  //     const text = decoder.decode(value);
  //     console.log("DECODE", done);
  //     console.log(text);

  //     if (text.status === "progress") {
  //       console.log("PROGRESS");
  //     }
  //     else if (text.status === "succeeded") {
  //       console.log("succeeded");
  //     }
  //     else {
  //       console.log("extra?")
  //     }

  //     console.log("-----------------");

  //     if (done as boolean) {
  //       reader.releaseLock();
  //       break;
  //     }
  //   }
  // };

  // useEffect(() => {
  //   const fetch = async () => {
  //     const res = await doMakeImage(options);
  //     void _handleStreamData(res);
  //   }

  //   if (isEnabled) {
  //     console.log('isEnabled');
  //     debugger;
  //     fetch()
  //       .catch((err) => {
  //         console.error(err);
  //       });
  //   }

  // }, [isEnabled, options, _handleStreamData]);

  // helper for the loading state to be enabled aware
  // useEffect(() => {
  //   if (isEnabled && status === "loading") {
  //     setIsLoading(true);
  //   } else {
  //     setIsLoading(false);
  //   }
  // }, [isEnabled, status]);


  // this is where there loading actually happens
  // useEffect(() => {
  //   console.log('DISPLATPANEL: status', status);
  //   console.log('DISPLATPANEL: data', data);

  //   // query is done
  //   if (status === "success") {
  //     // check to make sure that the image was created

  //     void _handleStreamData(data);

  //     // if (data.status === "succeeded") {
  //     //   if (isSoundEnabled) {
  //     //     // not awaiting the promise or error handling
  //     //     void dingRef.current?.play();
  //     //   }
  //     //   removeFirstInQueue();
  //     // }
  //   }
  // }, [status, data, removeFirstInQueue, dingRef, isSoundEnabled, _handleStreamData]);

  /* COMPLETED IMAGES */
  // const queryClient = useQueryClient();
  // const [completedImages, setCompletedImages] = useState<CompletedImagesType[]>(
  //   []
  // );

  // const completedIds = useImageQueue((state) => state.completedImageIds);
  // const clearCachedIds = useImageQueue((state) => state.clearCachedIds);

  // this is where we generate the list of completed images
  // useEffect(() => {
  //   const completedQueries = completedIds.map((id) => {
  //     const imageData = queryClient.getQueryData([MakeImageKey, id]);
  //     return imageData;
  //   }) as ImageReturnType[];

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

  //     setCompletedImages(temp);

  //     // could move this to the useEffect for completedImages
  //     if (temp.length > 0) {
  //       setCurrentImage(temp[0]);
  //     } else {
  //       setCurrentImage(null);
  //     }
  //   } else {
  //     setCompletedImages([]);
  //     setCurrentImage(null);
  //   }
  // }, [setCompletedImages, setCurrentImage, queryClient, completedIds]);

  // this is how we remove them
  // const removeImages = () => {
  //   completedIds.forEach((id) => {
  //     queryClient.removeQueries([MakeImageKey, id]);
  //   });
  //   clearCachedIds();
  // };

  // const [currentImage, setCurrentImage] = useState<CompletedImagesType | null>(
  //   null
  // );

  // const getCurrentImage = useImageDisplay((state) => state.getCurrentImage);
  // const images = useImageDisplay((state) => state.images);

  // useEffect(() => {
  //   if (images.length > 0) {
  //     debugger;
  //     const cur = getCurrentImage();
  //     console.log("cur", cur);
  //     setCurrentImage(cur);
  //   } else {
  //     setCurrentImage(null);
  //   }
  // }, [images, getCurrentImage]);

  //   useEffect(() => {
  //     console.log("images CHANGED");
  //     debugger;
  //     if (len) > 0) {
  //     // console.log("images", images);
  //     setCurrentImage(getCurrentImage());
  //   }
  // }, [len]);


  return (
    <div className={displayPanel}>
      DISPLAY
      {/* <AudioDing ref={dingRef}></AudioDing> */}
      <div className={displayContainer}>
        <CurrentDisplay
        ></CurrentDisplay>
      </div>

      <div className={previousImages}>
        <CompletedImages
        // removeImages={removeImages}
        // images={completedImages}
        // setCurrentDisplay={setCurrentImage}
        ></CompletedImages>
      </div>

    </div>
  );
}
