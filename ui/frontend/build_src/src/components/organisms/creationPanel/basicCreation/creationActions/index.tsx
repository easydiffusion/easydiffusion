import React from "react";

import MakeButton from "../../../../molecules/makeButton";

import ShowQueue from "../showQueue";

import {
  CreationActionMain
} from "./creationActions.css";

export default function CreationActions() {
  return (
    <div className={CreationActionMain}>
      <MakeButton></MakeButton>
      {/* <ShowQueue></ShowQueue> */}
    </div>
  );
}