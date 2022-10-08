import React, { useEffect, useState } from "react";
import TimeRemaining from "../../atoms/timeRemaining";

import { FetchingStates, useImageFetching } from "../../../stores/imageFetchingStore";
import { useRequestQueue } from "../../../stores/requestQueueStore";

export default function QueueStatusTab() {

  const [showTimeRemaining, setShowTimeRemaining] = useState(false);
  const [showQueueLength, setShowQueueLength] = useState(false);

  const status = useImageFetching((state) => state.status);

  const hasPendingQueue = useRequestQueue((state) => state.hasPendingQueue());
  const pendingRequests = useRequestQueue((state) => state.pendingRequests());

  useEffect(() => {
    console.log('status', status);
    // if we are processing we want to show something
    if (status == FetchingStates.PROGRESSING || status == FetchingStates.FETCHING) {
      // if we have a pending queue we want to show the queue length
      if (hasPendingQueue) {
        setShowTimeRemaining(false);
        setShowQueueLength(true);
      }
      // if we don't have a pending queue we want to show the time remaining
      else {
        setShowTimeRemaining(true);
        setShowQueueLength(false);
      }
    }
    else {
      setShowTimeRemaining(false);
      setShowQueueLength(false);
    }
  }, [status, hasPendingQueue]);

  return (
    <>
      <span>Queue </span>
      {showTimeRemaining && <span>: <TimeRemaining /></span>}
      {showQueueLength && <span> : {pendingRequests.length} remaining</span>}
    </>
  )

}



