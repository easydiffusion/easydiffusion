import React, { useEffect, useState } from "react";
import { useImageFetching } from '../../../stores/imageFetchingStore';

export default function TimeRemaining() {

  const step = useImageFetching((state) => state.step);
  const totalSteps = useImageFetching((state) => state.totalSteps);
  const startTime = useImageFetching((state) => state.timeStarted);
  const timeNow = useImageFetching((state) => state.timeNow);

  const [timeRemaining, setTimeRemaining] = useState('calculating...');

  useEffect(() => {
    // find the remaining time
    const timeTaken = +timeNow - +startTime;
    const timePerStep = step == 0 ? 0 : timeTaken / step;
    const totalTime = timePerStep * totalSteps;
    const timeRemaining = (totalTime - timeTaken) / 1000;
    if (timeRemaining < 1) {
      setTimeRemaining('calculating...');
    }

    setTimeRemaining(`${timeRemaining.toPrecision(3)} seconds remaining`);

  }, [step, totalSteps, startTime, timeNow, setTimeRemaining]);


  return (
    <>{timeRemaining}</>
  );
}