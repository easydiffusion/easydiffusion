import React, { useEffect, useState } from "react";
import { useImageQueue } from "../../../store/imageQueueStore";

import { doMakeImage, MakeImageKey } from "../../../api";

import { useQuery } from "@tanstack/react-query";
import GeneratedImage from "../generatedImage";


export default function CurrentImage() {

  const [imageData, setImageData] = useState(null);
  // @ts-ignore
  const {id, options} = useImageQueue((state) => state.firstInQueue());
  console.log('CurrentImage id', id)


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
    if(status === 'success') {
      console.log("success");

      // check to make sure that the image was created
      if(data.status === 'succeeded') {
        console.log("succeeded");
        setImageData(data.output[0].data);
        removeFirstInQueue();
      }
    }

  }, [status, data, removeFirstInQueue]);


  return (
    <div className="current-display">
      <h1>Current Image</h1>
      {imageData && <GeneratedImage imageData={imageData} />}
    </div>
  );
};