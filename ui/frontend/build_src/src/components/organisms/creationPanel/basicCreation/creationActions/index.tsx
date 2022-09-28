import React from "react";

import MakeButton from "../../../../atoms/makeButton";
import StopButton from "../../../../atoms/stopButton";
import ClearQueue from "../../../../atoms/clearQueue";

import {
  StopContainer
} from "./creationActions.css";

export default function CreationActions() {
  return (
    <div>
      <MakeButton></MakeButton>
      <div className={StopContainer}>
        <StopButton></StopButton>
        <ClearQueue></ClearQueue>
      </div>
    </div>
  );
}