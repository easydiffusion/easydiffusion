import React from "react";
import { doStopImage } from "../../../api";
import { useRequestQueue } from "../../../stores/requestQueueStore";

import {
  buttonStyle
} from "../../_recipes/button.css";


export default function ClearQueue() {

  const hasQueue = useRequestQueue((state) => state.hasAnyQueue());
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
    <button className={buttonStyle(
      {
        color: "cancel",
        size: "large",
      }
    )}
      disabled={!hasQueue} onClick={() => void stopAll()}>
      STOP ALL
    </button>
  );
}