import React, { useMemo, useState } from "react";
import { ImageRequest } from "../../../api/api.d";

import {
  QueueStatus,
  QueuedRequest,
  useRequestQueue
} from "../../../stores/requestQueueStore";

import {
  QueueDisplayMain,
  QueueListButtons,
} from "./queueDisplay.css";

import {
  buttonStyle
} from "../../_recipes/button.css";

import ClearQueue from "../../molecules/clearQueue";
import QueueItem from "./queueItem";

export default function QueueDisplay() {

  const requests: QueuedRequest[] = useRequestQueue((state) => state.requests);

  const [currentRequest, setCurrentRequest] = useState<QueuedRequest | null>(null);
  const [remainingRequests, setRemainingRequests] = useState<QueuedRequest[]>([]);

  const removeCompleted = useRequestQueue((state) => state.removeCompleted);
  const removeErrored = useRequestQueue((state) => state.removeErrored);

  const clearCompleted = () => {
    removeCompleted();
  }

  const clearErrored = () => {
    removeErrored();
  }

  useMemo(() => {
    const remaining = requests.filter((request) => {
      if (request.status != QueueStatus.processing) {
        return true
      }
    });

    const current = requests.find((request) => request.status == QueueStatus.processing);

    setCurrentRequest(current ?? null);
    setRemainingRequests(remaining);

  }, [requests]);

  return (
    <div className={QueueDisplayMain}>
      <ClearQueue />
      <div className={QueueListButtons}>
        <button
          className={buttonStyle({
            type: 'outline',
          })}
          onClick={clearCompleted}>Clear Completed</button>
        <button
          className={buttonStyle({
            type: 'outline',
          })}
          onClick={clearErrored}>Clear Errored</button>
      </div>

      {(currentRequest != null) && <QueueItem request={currentRequest} />}

      {remainingRequests.map((request) => {
        return <QueueItem key={request.batchId} request={request}></QueueItem>;
      })}
    </div>
  );
};
