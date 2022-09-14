import React, { useEffect, useState } from "react";

import { useImageCreate } from "../../../store/imageCreateStore";
import { useImageQueue } from "../../../store/imageQueueStore";
import { v4 as uuidv4 } from "uuid";

import { useRandomSeed } from "../../../utils";

export default function MakeButton() {
  const parallelCount = useImageCreate((state) => state.parallelCount);
  const builtRequest = useImageCreate((state) => state.builtRequest);
  const addNewImage = useImageQueue((state) => state.addNewImage);
  const isRandomSeed = useImageCreate((state) => state.isRandomSeed());
  const setRequestOption = useImageCreate((state) => state.setRequestOptions);

  const makeImages = () => {
    // the request that we have built
    const req = builtRequest();
    // the actual number of request we will make
    let requests = [];
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
        seed: seed,
      });
    });

    // potentially update the seed
    if (isRandomSeed) {
      // update the seed for the next time we click the button
      setRequestOption("seed", useRandomSeed());
    }

  };

  return <button onClick={makeImages}>Make</button>;
}
