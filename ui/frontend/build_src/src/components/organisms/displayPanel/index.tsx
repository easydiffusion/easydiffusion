
import React from "react";

import CurrentDisplay from "./currentDisplay";
import CompletedImages from "./completedImages";

import {
  displayPanel,
  displayContainer,
  previousImages,
} from "./displayPanel.css";

export default function DisplayPanel() {

  return (
    <div className={displayPanel}>

      <div className={displayContainer}>
        <CurrentDisplay
        ></CurrentDisplay>
      </div>

      <div className={previousImages}>
        <CompletedImages
        ></CompletedImages>
      </div>

    </div>
  );
}
