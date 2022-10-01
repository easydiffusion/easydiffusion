
import React, { useState } from "react";

import CurrentDisplay from "./currentDisplay";
import CompletedImages from "./completedImages";

// import {
//   tabStyles
// } from "../../_recipes/tabs_headless.css";
import {
  displayPanel,
  // displayContainer,
} from "./displayPanel.css";



export default function DisplayPanel() {

  // const [isShowing, setIsShowing] = useState(false)

  return (
    <div className={displayPanel}>

      {/* <div className={displayContainer}> */}
      <CurrentDisplay
      ></CurrentDisplay>
      {/* </div> */}

      {/* <div className={previousImages}> */}
      <CompletedImages
      ></CompletedImages>
      {/* </div> */}
    </div>
  );
}
