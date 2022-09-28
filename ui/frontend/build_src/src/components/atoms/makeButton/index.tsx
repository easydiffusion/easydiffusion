/* eslint-disable @typescript-eslint/naming-convention */
import React, { useEffect, useRef } from "react";

import { useImageCreate } from "../../../stores/imageCreateStore";
import { useImageQueue } from "../../../stores/imageQueueStore";
import {
  FetchingStates,
  useImageFetching
} from "../../../stores/imageFetchingStore";


import { useImageDisplay } from "../../../stores/imageDisplayStore";

import { v4 as uuidv4 } from "uuid";

import { useRandomSeed } from "../../../utils";
import {
  ImageRequest,
  ImageReturnType,
  ImageOutput,
  doMakeImage,
} from "../../../api";
import {
  MakeButtonStyle,
} from "./makeButton.css";

import { useTranslation } from "react-i18next";

import AudioDing from "../../molecules/audioDing";

const idDelim = "_batch";

export default function MakeButton() {
  const { t } = useTranslation();

  const dingRef = useRef<HTMLAudioElement>();

  const parallelCount = useImageCreate((state) => state.parallelCount);
  const builtRequest = useImageCreate((state) => state.builtRequest);
  const isRandomSeed = useImageCreate((state) => state.isRandomSeed());
  const setRequestOption = useImageCreate((state) => state.setRequestOptions);

  const isSoundEnabled = useImageCreate((state) => state.isSoundEnabled());

  const addNewImage = useImageQueue((state) => state.addNewImage);
  const hasQueue = useImageQueue((state) => state.hasQueuedImages());
  const removeFirstInQueue = useImageQueue((state) => state.removeFirstInQueue);
  const { id, options } = useImageQueue((state) => state.firstInQueue());

  const status = useImageFetching((state) => state.status);
  const setStatus = useImageFetching((state) => state.setStatus);
  const setStep = useImageFetching((state) => state.setStep);
  const setTotalSteps = useImageFetching((state) => state.setTotalSteps);
  const addProgressImage = useImageFetching((state) => state.addProgressImage);
  const setStartTime = useImageFetching((state) => state.setStartTime);
  const setNowTime = useImageFetching((state) => state.setNowTime);
  const resetForFetching = useImageFetching((state) => state.resetForFetching);
  const appendData = useImageFetching((state) => state.appendData);

  const updateDisplay = useImageDisplay((state) => state.updateDisplay);

  const hackJson = (jsonStr: string, id: string) => {

    try {
      const parsed = JSON.parse(jsonStr);
      const { status, request, output: outputs } = parsed as ImageReturnType;
      if (status === 'succeeded') {
        outputs.forEach((output: any, index: number) => {

          const { data, seed } = output as ImageOutput;
          const seedReq = {
            ...request,
            seed,
          };
          const batchId = `${id}${idDelim}-${seed}-${index}`;
          updateDisplay(batchId, data, seedReq);
        });
      }

      else {
        console.warn(`Unexpected status: ${status}`);
      }

    }
    catch (e) {
      console.log("Error HACKING JSON: ", e)
    }
  }

  const parseRequest = async (id: string, reader: ReadableStreamDefaultReader<Uint8Array>) => {
    const decoder = new TextDecoder();
    let finalJSON = '';

    while (true) {
      const { done, value } = await reader.read();
      const jsonStr = decoder.decode(value);
      if (done) {
        setStatus(FetchingStates.COMPLETE);
        hackJson(finalJSON, id);
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
              // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
              const timePath = `${output.path}?t=${new Date().getTime()}`
              addProgressImage(timePath);
            });
          }

        } else if (status === "succeeded") {
          // TODO this should be the the new out instead of the try catch
          // wait for the path to come back instead of the data
          setStatus(FetchingStates.SUCCEEDED);
          console.log(update);
        }
        else if (status === 'failed') {
          console.warn('failed');
          console.log(update);
        }
        else {
          console.log("UNKNOWN ?", update);
        }
      }
      catch (e) {
        console.log('EXPECTED PARSE ERRROR')
        finalJSON += jsonStr;
      }

    }
  }

  const startStream = async (id: string, req: ImageRequest) => {


    try {
      resetForFetching();
      const res = await doMakeImage(req);
      const reader = res.body?.getReader();

      if (void 0 !== reader) {
        void parseRequest(id, reader);
      }

    } catch (e) {
      console.log('TOP LINE STREAM ERROR')
      console.log(e);
    }

  }

  const queueImageRequest = (req: ImageRequest) => {
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
      addNewImage(uuidv4(), {
        ...req,
        // updated the number of images to make
        num_outputs: num,
        // update the seed
        seed,
      });
    });
  }

  const makeImageQueue = async () => {
    // potentially update the seed
    if (isRandomSeed) {
      // update the seed for the next time we click the button
      setRequestOption("seed", useRandomSeed());
    }
    // the request that we have built
    const req = builtRequest();
    queueImageRequest(req);
  };

  useEffect(() => {
    const makeImages = async (options: ImageRequest) => {
      removeFirstInQueue();
      await startStream(id ?? "", options);
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

  }, [hasQueue, status, id, options, startStream]);

  return (
    <>
      <button
        className={MakeButtonStyle}
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
