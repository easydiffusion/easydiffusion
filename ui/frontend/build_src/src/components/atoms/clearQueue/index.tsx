import React from "react";
import { doStopImage } from "../../../api";
import { useImageQueue } from "../../../stores/imageQueueStore";
import { BrandedButton } from "../../../styles/shared.css";

import { useCreateUI } from "../../../components/organisms/creationPanel/creationPanelUIStore";

export default function ClearQueue() {

  const hasQueue = useImageQueue((state) => state.hasQueuedImages());
  const clearQueue = useImageQueue((state) => state.clearQueue);

  const showQueue = useCreateUI((state) => state.showQueue);
  const toggleQueue = useCreateUI((state) => state.toggleQueue);


  const stopAll = async () => {
    console.log("stopAll");
    try {
      clearQueue();
      const res = await doStopImage();
    } catch (e) {
      console.log(e);
    }
  };

  return (
    <>
      <button className={BrandedButton} disabled={!hasQueue} onClick={() => void stopAll()}>Clear Queue</button>
      <label>
        <input
          type="checkbox"
          checked={showQueue}
          onChange={() => toggleQueue()}
        >
        </input>
        Display
      </label>
    </>
  );
}