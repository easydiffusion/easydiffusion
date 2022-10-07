
import React, { useState, useRef, useEffect } from "react";
import { Transition } from '@headlessui/react'

import CurrentDisplay from "../../organisms/currentDisplay";
import CompletedImages from "../../organisms/completedImages";
import CurrentInfo from "../../organisms/currentInfo";


import {
  tabStyles
} from "../../_recipes/tabs_headless.css";

import {
  basicDisplayLayout,
  contentLayout,
  infoLayout,
  historyLayout
} from "./basicDisplay.css";

export default function BasicDisplay() {

  const [isShowingHistory, setIsShowingHistory] = useState(true)

  const layoutRef = useRef<HTMLDivElement>(null);

  useEffect(() => {

    if (layoutRef.current != undefined) {
      // set the hide-history data attribute
      if (isShowingHistory) {
        layoutRef.current.removeAttribute('data-hide-history');
      }
      else {
        // layoutRef.current.dataset.hideHistory = "true";
        layoutRef.current.setAttribute('data-hide-history', '');
      }
    }
  }, [layoutRef, isShowingHistory]);


  return (
    <div
      ref={layoutRef}
      className={basicDisplayLayout}
    >
      <div className={contentLayout}>
        <CurrentDisplay></CurrentDisplay>
      </div>

      {/* <div className={infoLayout}>
        <CurrentInfo ></CurrentInfo>
      </div> */}

      <div className={historyLayout}>
        <button
          className={tabStyles({})}
          onClick={() => setIsShowingHistory((isShowingHistory) => !isShowingHistory)}>
          {isShowingHistory ? "Hide History" : "Show History"}
        </button>
        <CompletedImages></CompletedImages>
      </div>
    </div>
  );
}
