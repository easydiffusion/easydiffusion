import React from "react";

import MakeButton from "../../../../molecules/makeButton";
// import StopButton from "../../../../molecules/stopButton";
// import ClearQueue from "../../../../molecules/clearQueue";

import ShowQueue from "../showQueue";

import {
  StopContainer
} from "./creationActions.css";

export default function CreationActions() {
  return (
    <div>
      <MakeButton></MakeButton>
      <ShowQueue></ShowQueue>
      {/* <div className={StopContainer}>
        <StopButton></StopButton>
        <ClearQueue></ClearQueue>
      </div> */}
    </div>
  );
}