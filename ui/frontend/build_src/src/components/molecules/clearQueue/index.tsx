import React from "react";
import { doStopImage } from "../../../api";
import { useRequestQueue } from "../../../stores/requestQueueStore";

import {
  ClearQueueButton
} from "./clearQueue.css";

export default function ClearQueue() {

  const hasQueue = useRequestQueue((state) => state.hasPendingQueue());
  const clearQueue = useRequestQueue((state) => state.clearQueue);

  const stopAll = async () => {
    try {
      clearQueue();
      const res = await doStopImage();
    } catch (e) {
      console.log(e);
    }
  };

  return (
    <button className={ClearQueueButton} disabled={!hasQueue} onClick={() => void stopAll()}>
      STOP ALL
    </button>
  );
}