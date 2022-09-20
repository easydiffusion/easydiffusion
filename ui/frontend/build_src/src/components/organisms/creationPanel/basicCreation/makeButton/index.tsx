/* eslint-disable @typescript-eslint/naming-convention */
import React from "react";

import { useImageCreate } from "../../../../../stores/imageCreateStore";
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

  const makeImages = () => {

    // potentially update the seed
    if (isRandomSeed) {
      // update the seed for the next time we click the button
      debugger;
      setRequestOption("seed", useRandomSeed());
    }

    // the request that we have built
    const req = builtRequest();
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
        debugger;
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
