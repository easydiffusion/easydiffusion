import React, {useEffect, useState}from "react";

import { useImageCreate } from "../../../store/imageCreateStore";
// import { useImageDisplay } from "../../../store/imageDisplayStore";
import { useImageQueue } from "../../../store/imageQueueStore"; 
// import { doMakeImage } from "../../../api"; 
import {v4 as uuidv4} from 'uuid';

export default function MakeButton() {

  const builtRequest = useImageCreate((state) => state.builtRequest);
  const addNewImage = useImageQueue((state) => state.addNewImage);
  
  const makeImage = () => {
    // todo turn this into a loop and adjust the parallel count
    // 
    const req =  builtRequest();
    addNewImage(uuidv4(), req)
  };
  
  return (
     <button onClick={makeImage}>Make</button>
  );
}