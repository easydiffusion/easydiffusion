/* eslint-disable @typescript-eslint/no-unnecessary-type-assertion */
/* eslint-disable @typescript-eslint/prefer-ts-expect-error */
/* eslint-disable @typescript-eslint/naming-convention */
import React, { useEffect } from "react";

import { useImageCreate, ImageRequest } from "../../../../../stores/imageCreateStore";
import { useImageQueue } from "../../../../../stores/imageQueueStore";
import {
  FetchingStates,
  useImageFetching
} from "../../../../../stores/imageFetchingStore";

import { v4 as uuidv4 } from "uuid";

import { useRandomSeed } from "../../../../../utils";
import { doMakeImage } from "../../../../../api";
import {
  MakeButtonStyle, // @ts-expect-error
} from "./makeButton.css.ts";

import { useTranslation } from "react-i18next";

import AudioDing from "./audioDing";
import { parse } from "node:path/win32";

export default function MakeButton() {
  const { t } = useTranslation();

  const parallelCount = useImageCreate((state) => state.parallelCount);
  const builtRequest = useImageCreate((state) => state.builtRequest);
  const isRandomSeed = useImageCreate((state) => state.isRandomSeed());
  const setRequestOption = useImageCreate((state) => state.setRequestOptions);

  const addNewImage = useImageQueue((state) => state.addNewImage);
  const hasQueue = useImageQueue((state) => state.hasQueuedImages());
  const { id, options } = useImageQueue((state) => state.firstInQueue());

  const setStatus = useImageFetching((state) => state.setStatus);
  const appendData = useImageFetching((state) => state.appendData);

  const parseRequest = async (reader: ReadableStreamDefaultReader<Uint8Array>) => {
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();

      if (done as boolean) {
        console.log("DONE");
        setStatus(FetchingStates.COMPLETE);
        break;
      }

      const jsonStr = decoder.decode(value);

      try {
        const update = JSON.parse(jsonStr);

        if (update.status === "progress") {
          console.log("PROGRESS");
          setStatus(FetchingStates.PROGRESSING);
        }
        else if (update.status === "succeeded") {
          console.log("succeeded");
          setStatus(FetchingStates.SUCCEEDED);
          // appendData(update.data);
        }
        else {
          console.log("extra?", update);
          // appendData(update.data);
        }
      }
      catch (e) {
        console.log('PARSE ERRROR')
        console.log(e)
        debugger;
        //    appendData(update.data);
      }

    }
  }

  const startStream = async (req: ImageRequest) => {

    const streamReq = {
      ...req,
      // stream_image_progress: false,
    };

    console.log("testStream", streamReq);
    try {
      const res = await doMakeImage(streamReq);
      // @ts-expect-error
      const reader = res.body.getReader();
      void parseRequest(reader);

    } catch (e) {
      console.log('e');
    }

  }

  const queueImageRequest = async (req: ImageRequest) => {
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
    await queueImageRequest(req);
    // void startStream(req);
  };

  useEffect(() => {

    const makeImages = async (options: ImageRequest) => {
      // potentially update the seed
      await startStream(options);
    }

    if (hasQueue) {
      makeImages(options).catch((e) => {
        console.log('HAS QUEUE ERROR');
        console.log(e);
      });
    }


  }, [hasQueue, id, options, startStream]);

  return (
    <button
      className={MakeButtonStyle}
      onClick={() => {
        setStatus(FetchingStates.FETCHING);
        void makeImageQueue();
      }}
      disabled={hasQueue}
    >
      {t("home.make-img-btn")}
    </button>
  );
}
