import React, {useEffect, useState}from "react";

import { useImageCreate } from "../../../store/imageCreateStore";
import { useImageQueue } from "../../../store/imageQueueStore"; 
import {v4 as uuidv4} from 'uuid';

import { useRandomSeed } from "../../../utils";

export default function MakeButton() {

  const parallelCount = useImageCreate((state) => state.parallelCount);
  const builtRequest = useImageCreate((state) => state.builtRequest);
  const addNewImage = useImageQueue((state) => state.addNewImage);
  
  const makeImages = () => {
    const req = builtRequest();
    let requests = [];
    let { num_outputs } = req;

    if( parallelCount > num_outputs ) {
      requests.push(num_outputs);
    }
    
    else {
      while (num_outputs >= 1) {
        num_outputs -= parallelCount;
        if(num_outputs <= 0) {
          requests.push(parallelCount)
        }
        else {
          requests.push(Math.abs(num_outputs))
        }
      }
    }

    console.log('requests', requests);

    requests.forEach((num, index) => {

      console.log('num', num);
      let seed = req.seed;
      if(index !== 0) {
        seed = useRandomSeed();
      }

      // debugger;

      addNewImage(uuidv4(), {
        ...req, 
        num_outputs: num,
        seed: seed
      })
    });
    
  };
  
  return (
     <button onClick={makeImages}>Make</button>
  );
}