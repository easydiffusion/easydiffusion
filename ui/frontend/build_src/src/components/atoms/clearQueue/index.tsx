import React from "react";
import { doStopImage } from "../../../api";
import { useImageQueue } from "../../../stores/imageQueueStore";
import { BrandedButton } from "../../../styles/shared.css";

export default function ClearQueue() {

  const hasQueue = useImageQueue((state) => state.hasQueuedImages());
  const clearQueue = useImageQueue((state) => state.clearQueue);

  const stopAll = async () => {
    console.log("stopAll");
    try {
      clearQueue();
      const res = await doStopImage();
    } catch (e) {
      console.log(e);
    }
  };
  // / disabled={!hasQueue}

  return (
    <button className={BrandedButton} disabled={!hasQueue} onClick={() => void stopAll()}>Clear Queue</button>
  );
}