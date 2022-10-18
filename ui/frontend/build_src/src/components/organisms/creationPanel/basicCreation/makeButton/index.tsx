/* eslint-disable @typescript-eslint/naming-convention */
import React, { useEffect, useRef } from "react";

import { useImageCreate } from "../../../../../stores/imageCreateStore";

import {
  usePromptMatrix
} from "../../../../../stores/promptMatrixStore";

import {
  QueueStatus,
  useRequestQueue
} from "../../../../../stores/requestQueueStore";

import {
  FetchingStates,
  useImageFetching
} from "../../../../../stores/imageFetchingStore";
import { useProgressImages } from "../../../../../stores/progressImagesStore";
import { useCreatedMedia } from "../../../../../stores/createdMediaStore";
import { useImageDisplay } from "../../../../../stores/imageDisplayStore";

import { v4 as uuidv4 } from "uuid";

import { useRandomSeed } from "../../../../../utils";
import {
  ImageRequest,
  ImageReturnType,
  ImageOutput,
} from "../../../../../api/api";

import {
  doMakeImage,
} from "../../../../../api";

import {
  buttonStyle
} from "../../../../_recipes/button.css";

import { useTranslation } from "react-i18next";

import AudioDing from "../../../../molecules/_stateless/audioDing";

const idDelim = "_item";

interface Props {
  className?: string;
}


export default function MakeButton({ className }: Props) {
  const { t } = useTranslation();

  const dingRef = useRef<HTMLAudioElement>();

  // creation logic
  const parallelCount = useImageCreate((state) => state.parallelCount);
  const builtRequest = useImageCreate((state) => state.builtRequest);
  const isRandomSeed = useImageCreate((state) => state.isRandomSeed());
  const setRequestOption = useImageCreate((state) => state.setRequestOptions);
  const isSoundEnabled = useImageCreate((state) => state.isSoundEnabled());

  // request modify logic
  const promptsList = usePromptMatrix((state) => state.getSafeList());

  const shouldClearOnCreate = usePromptMatrix((state) => state.shouldClearOnCreate);
  const clearPromptMatrix = usePromptMatrix((state) => state.clearPromptMatrix);

  // request queue logic
  const addtoQueue = useRequestQueue((state) => state.addtoQueue);
  const hasQueue = useRequestQueue((state) => state.hasPendingQueue());
  const { batchId, options } = useRequestQueue((state) => state.firstInQueue());
  const updateQueueStatus = useRequestQueue((state) => state.updateStatus);

  // fetching logic
  const status = useImageFetching((state) => state.status);
  const setStatus = useImageFetching((state) => state.setStatus);
  const setStep = useImageFetching((state) => state.setStep);
  const setTotalSteps = useImageFetching((state) => state.setTotalSteps);
  const setStartTime = useImageFetching((state) => state.setStartTime);
  const setNowTime = useImageFetching((state) => state.setNowTime);
  const resetForFetching = useImageFetching((state) => state.resetForFetching);

  // progress images logic
  const addProgressImage = useProgressImages((state) => state.addProgressImage);

  // created logic
  const removeFailedMedia = useCreatedMedia((state) => state.removeFailedMedia);
  const addCreatedMedia = useCreatedMedia((state) => state.addCreatedMedia);

  // display logic
  const shouldDisplayWhenComplete = useImageDisplay((state) => state.shouldDisplayWhenComplete);
  const setCurrentImage = useImageDisplay((state) => state.setCurrentImage);


  const hackJson = (jsonStr: string, batchId: string) => {

    try {
      const parsed = JSON.parse(jsonStr);
      const { status, request, output: outputs } = parsed as ImageReturnType;

      if (status === 'succeeded') {

        updateQueueStatus(batchId, QueueStatus.complete);
        outputs.forEach((output: any, index: number) => {

          const { data, seed } = output as ImageOutput;
          const seedReq = {
            ...request,
            seed,
          };
          const itemId = `${batchId}${idDelim}-${seed}-${index}`;
          // updateDisplay(batchId, data, seedReq);
          addCreatedMedia(batchId, seed, seedReq, { id: itemId, data });


          if (shouldDisplayWhenComplete) {
            setCurrentImage({ batchId, imageId: itemId, seed });
          }

          // const setCurrentImage = useImageDisplay((state) => state.setCurrentImage);
          // const setProgressAsCurrent = (progressId: string) => {
          //   console.log('setProgressAsCurrent - batchId', batchId);
          //   console.log('progressId', progressId);
          //   if (batchId != null && seed != null) {
          //     setCurrentImage({ batchId, progressId, seed });
          //   }
          // }

        });
      }

      else {
        console.warn(`Unexpected status: ${status}`);
        updateQueueStatus(batchId, QueueStatus.error);
      }

    }
    catch (e) {
      updateQueueStatus(batchId, QueueStatus.error);
      console.warn("Error HACKING JSON: ", e)
    }
  }

  const parseRequest = async (batchId: string, reader: ReadableStreamDefaultReader<Uint8Array>) => {
    const decoder = new TextDecoder();
    let finalJSON = '';


    while (true) {
      const { done, value } = await reader.read();
      const jsonStr = decoder.decode(value);

      if (done) {
        setStatus(FetchingStates.COMPLETE);
        hackJson(finalJSON, batchId);
        if (isSoundEnabled) {
          void dingRef.current?.play();
        }
        break;
      }

      try {
        const update = JSON.parse(jsonStr);
        const { status } = update;

        if (status === "progress") {
          setStatus(FetchingStates.PROGRESSING);
          const { progress: { step, total_steps }, output: outputs } = update;
          setStep(step);
          setTotalSteps(total_steps);

          if (step === 0) {
            setStartTime();
          }
          else {
            setNowTime();
          }

          if (void 0 !== outputs) {
            outputs.forEach((output: any) => {
              const { path, seed } = output;
              console.log('output path', path);
              // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
              const timePath = `${path}?t=${new Date().getTime()}`
              console.log('output path', timePath);
              addProgressImage(batchId, seed, { id: uuidv4(), data: timePath });
            });
          }

        } else if (status === "succeeded") {
          // TODO this should be the the new out instead of the try catch
          // wait for the path to come back instead of the data
          setStatus(FetchingStates.SUCCEEDED);
        }
        else if (status === 'failed') {
          console.warn('failed');
          console.warn(update);
        }
        else {
          console.warn("UNKNOWN ?", update);
        }
      }
      catch (e) {
        // console.log('EXPECTED PARSE ERRROR')
        finalJSON += jsonStr;
      }
    }
  }

  const startStream = async (batchId: string, req: ImageRequest) => {

    try {
      updateQueueStatus(batchId, QueueStatus.processing);
      resetForFetching();
      const res = await doMakeImage(req);
      const reader = res.body?.getReader();

      if (void 0 !== reader) {
        // makeSpace(batchId, req);
        void parseRequest(batchId, reader);
      }

    } catch (e) {
      console.log('TOP LINE STREAM ERROR')
      updateQueueStatus(batchId, QueueStatus.error);
      removeFailedMedia(batchId);
      console.log(e);
    }
  }

  const queueImageRequest = (req: ImageRequest) => {

    promptsList.forEach((prompt) => {

      // marry the modifiers to the prompt
      const { options } = prompt;
      // TODO clean up some of the comma logic
      const positivePrompt = options.filter((t) => t.type === "positive").map((t) => t.name).join(",");
      const negativePrompt = options.filter((t) => t.type === "negative").map((t) => t.name).join(",");
      const fullPrompt = `${req.prompt}, ${positivePrompt}`;
      const fullNegativePrompt = `${req.negative_prompt}, ${negativePrompt}`;
      req.prompt = fullPrompt;
      req.negative_prompt = fullNegativePrompt;

      // the actual number of request we will make
      const requests = [];
      // the number of images we will make
      let { num_outputs } = req;
      if (parallelCount > num_outputs) {
        requests.push(num_outputs);
      } else {
        // while we have at least 1 image to make
        while (num_outputs >= 1) {
          // subtract the parallel count from the number of images to make
          num_outputs -= parallelCount;

          // if we are still 0 or greater we can make the full parallel count
          if (num_outputs <= 0) {
            requests.push(parallelCount);
          }
          // otherwise we can only make the remaining images
          else {
            requests.push(Math.abs(num_outputs));
          }
        }
      }

      requests.forEach((num, index) => {
        // get the seed we want to use
        let seed = req.seed;
        if (index !== 0) {
          // we want to use a random seed for subsequent requests
          seed = useRandomSeed();
        }
        // add the request to the queue
        addtoQueue(uuidv4(), {
          ...req,
          // updated the number of images to make
          num_outputs: num,
          // update the seed
          seed,
        });
      });
    });

    if (shouldClearOnCreate) {
      clearPromptMatrix();
    }
  }

  const makeImageQueue = async () => {
    // potentially update the seed
    if (isRandomSeed) {
      // update the seed for the next time we click the button
      setRequestOption("seed", useRandomSeed());
    }
    // the request that we have built
    const req = builtRequest();
    console.log('req', req);
    debugger;
    queueImageRequest(req);
  };

  useEffect(() => {
    const makeImages = async (options: ImageRequest) => {
      // removeFirstInQueue();
      await startStream(batchId ?? "", options);
    }

    if (status === FetchingStates.PROGRESSING || status === FetchingStates.FETCHING) {
      return;
    }

    if (hasQueue) {

      if (options === undefined) {
        console.log('req is undefined');
        return;
      }

      makeImages(options).catch((e) => {
        console.log('HAS QUEUE ERROR');
        console.log(e);
      });
    }

  }, [hasQueue, status, batchId, options, startStream]);

  return (
    <>
      <button
        className={[className, buttonStyle({
          size: 'large',
        })].join(" ")}
        onClick={() => {
          void makeImageQueue();
        }}
      >
        {t("home.make-img-btn")}
      </button>
      <AudioDing ref={dingRef}></AudioDing>
    </>
  );
}
