import React, { useEffect, useState } from "react";

import { FetchingStates, useImageFetching } from "../../../stores/imageFetchingStore";
import { useRequestQueue } from "../../../stores/requestQueueStore";

export default function QueueStatusTab() {


  const [showBasicQueue, setShowBasicQueue] = useState(true);

  const hasPendingQueue = useRequestQueue((state) => state.hasPendingQueue());
  const pendingRequests = useRequestQueue((state) => state.pendingRequests());

  const status = useImageFetching((state) => state.status);

  const step = useImageFetching((state) => state.step);
  const totalSteps = useImageFetching((state) => state.totalSteps);

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

  useEffect(() => {
    if (hasPendingQueue) {
      setShowBasicQueue(false);
    }
  }, [status, hasPendingQueue]);

  // {/* {showBasicQueue
  //   ? <> */}
  //   Queue
  //   {/* </>
  //   : <>
  //     <span>Percent: {percent}%</span>
  //   </>npm
  // } */}


  return (
    <>
      <span>Queue </span>
      {hasPendingQueue && <span> Items Remaining: {pendingRequests.length} </span>}
    </>
  )

}



