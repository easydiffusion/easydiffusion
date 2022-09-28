import React, { useEffect, useState } from "react";

import { FetchingStates, useImageFetching } from "../../../../stores/imageFetchingStore";
import { useImageDisplay } from "../../../../stores/imageDisplayStore";

import { API_URL } from "../../../../api";

import {
  currentDisplayMain,
} from './currentDisplay.css';

import ImageDisplay from "./imageDisplay";

const IdleDisplay = () => {
  return (
    <h4 className="no-image">Try Making a new image!</h4>
  );
};

const LoadingDisplay = () => {

  const step = useImageFetching((state) => state.step);
  const totalSteps = useImageFetching((state) => state.totalSteps);
  const progressImages = useImageFetching((state) => state.progressImages);

  const startTime = useImageFetching((state) => state.timeStarted);
  const timeNow = useImageFetching((state) => state.timeNow);
  const [timeRemaining, setTimeRemaining] = useState(0);

  const [percent, setPercent] = useState(0);


  useEffect(() => {
    if (totalSteps > 0) {
      setPercent(Math.round((step / totalSteps) * 100));
    } else {
      setPercent(0);
    }
  }, [step, totalSteps]);

  useEffect(() => {
    // find the remaining time
    const timeTaken = +timeNow - +startTime;
    const timePerStep = step == 0 ? 0 : timeTaken / step;
    const totalTime = timePerStep * totalSteps;
    const timeRemaining = (totalTime - timeTaken) / 1000;
    // @ts-expect-error
    setTimeRemaining(timeRemaining.toPrecision(3));

  }, [step, totalSteps, startTime, timeNow, setTimeRemaining]);

  return (
    <>
      <h4 className="loading">Loading...</h4>
      <p>{percent} % Complete </p>
      {timeRemaining != 0 && <p>Time Remaining: {timeRemaining} s</p>}
      {progressImages.map((image, index) => {
        if (index == progressImages.length - 1) {
          return (
            <img src={`${API_URL}${image}`} key={index} />
          )
        }
      })
      }
    </>
  );
};


export default function CurrentDisplay() {

  const status = useImageFetching((state) => state.status);
  const currentImage = useImageDisplay((state) => state.currentImage);

  return (
    <div className={currentDisplayMain}>

      {status === FetchingStates.IDLE && <IdleDisplay />}

      {(status === FetchingStates.FETCHING || status === FetchingStates.PROGRESSING) && <LoadingDisplay />}

      {(status === FetchingStates.COMPLETE && currentImage != null) && <ImageDisplay info={currentImage?.info} data={currentImage?.data} />}

    </div>
  );
}