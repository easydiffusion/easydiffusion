
import React, { useState, useRef, useEffect } from "react";


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
  infoTab,
  historyLayout
} from "./basicDisplay.css";

export default function BasicDisplay() {

  const [isShowingHistory, setIsShowingHistory] = useState(true);
  const [isShowingInfo, setIsShowingInfo] = useState(true);

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

  useEffect(() => {
    if (layoutRef.current != undefined) {
      // set the hide-history data attribute
      if (isShowingInfo) {
        layoutRef.current.removeAttribute('data-hide-info');
      }
      else {
        // layoutRef.current.dataset.hideHistory = "true";
        layoutRef.current.setAttribute('data-hide-info', '');
      }
    }
  }, [layoutRef, isShowingInfo]);


  return (
    <div
      ref={layoutRef}
      className={basicDisplayLayout}
    >
      <div className={contentLayout}>
        <CurrentDisplay></CurrentDisplay>
      </div>

      <div className={infoLayout}>
        <button
          className={infoTab}
          onClick={() => setIsShowingInfo((isShowingInfo) => !isShowingInfo)}>
          {isShowingInfo ? "Toggle info" : "Toggle info"}
        </button>
        <CurrentInfo ></CurrentInfo>
      </div>

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
