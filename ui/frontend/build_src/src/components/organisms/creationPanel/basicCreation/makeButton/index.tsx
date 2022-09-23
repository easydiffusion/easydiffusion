/* eslint-disable @typescript-eslint/naming-convention */
import React from "react";

import { useImageCreate, ImageRequest } from "../../../../../stores/imageCreateStore";
import { useImageQueue } from "../../../../../stores/imageQueueStore";
import { v4 as uuidv4 } from "uuid";

import { useRandomSeed } from "../../../../../utils";

import {
  MakeButtonStyle, // @ts-expect-error
} from "./makeButton.css.ts";

import { useTranslation } from "react-i18next";

export default function MakeButton() {
  const { t } = useTranslation();

  const parallelCount = useImageCreate((state) => state.parallelCount);
  const builtRequest = useImageCreate((state) => state.builtRequest);
  const addNewImage = useImageQueue((state) => state.addNewImage);
  const hasQueue = useImageQueue((state) => state.hasQueuedImages());
  const isRandomSeed = useImageCreate((state) => state.isRandomSeed());
  const setRequestOption = useImageCreate((state) => state.setRequestOptions);

  // const makeImages = () => {
  //   // potentially update the seed
  //   if (isRandomSeed) {
  //     // update the seed for the next time we click the button
  //     setRequestOption("seed", useRandomSeed());
  //   }

  //   // the request that we have built
  //   const req = builtRequest();

  // };

  const queueImageRequest = (req: ImageRequest) => {

    // the actual number of request we will make
    const requests = [];
    // the number of images we will make
    let { num_outputs } = req;

    // if making fewer images than the parallel count
    // then it is only 1 request
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

    // make the requests
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

  const testStream = async (req: ImageRequest) => {

    const streamReq = {
      ...req,
      stream_progress_updates: true,
      // stream_image_progress: false,
      session_id: uuidv4(),
    };

    console.log("testStream", streamReq);
    try {
      const res = await fetch('http://localhost:9000/image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(streamReq)
      });

      console.log('res', res);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        const text = decoder.decode(value);
        console.log(text);
      }
    } catch (e) {
      console.log(e);
      debugger;
    }

  }

  const makeImages = () => {
    // potentially update the seed
    if (isRandomSeed) {
      // update the seed for the next time we click the button
      setRequestOption("seed", useRandomSeed());
    }

    // the request that we have built
    const req = builtRequest();

    //queueImageRequest(req);
    void testStream(req);

  };


  return (
    <button
      className={MakeButtonStyle}
      onClick={makeImages}
      disabled={hasQueue}
    >
      {t("home.make-img-btn")}
    </button>
  );
}
