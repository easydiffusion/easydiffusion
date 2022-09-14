import React, { useEffect, useState } from "react";
import { useImageQueue } from "../../../store/imageQueueStore";

import { doMakeImage, MakeImageKey } from "../../../api";

import { useQuery } from "@tanstack/react-query";
import GeneratedImage from "../generatedImage";

// TODO move this logic to the display panel
export default function CurrentImage() {
  // const [imageData, setImageData] = useState(null);
  // @ts-ignore
  const { id, options } = useImageQueue((state) => state.firstInQueue());

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
    if (status === "success") {
      // check to make sure that the image was created
      if (data.status === "succeeded") {
        setImageData(data.output[0].data);
        removeFirstInQueue();
      }
    }
  }, [status, data, removeFirstInQueue]);

  return (
    <></>
    // <div className="current-display">
    //   <h1>Current Image</h1>
    //   {imageData && <GeneratedImage imageData={imageData} />}
    // </div>
  );
}
