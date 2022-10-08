import React from "react";
import { doStopImage } from "../../../api";


import {
  buttonStyle
} from "../../_recipes/button.css";

export default function StopButton() {

  const stopMake = async () => {
    try {
      const res = await doStopImage();
    } catch (e) {
      console.log(e);
    }
  };

  return <button className={buttonStyle(
    {
      color: "cancel",
      size: "large",
    }
  )} onClick={() => void stopMake()}>Stop</button>;
}
