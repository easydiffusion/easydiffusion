import React from "react";
import { doStopImage } from "../../../../../api";


export default function StopButton() {

  const stopMake = async () => {
    try {
      const res = await doStopImage();
    } catch (e) {
      console.log(e);
    }
  };

  return <button onClick={() => void stopMake()}>Stop</button>;
}
