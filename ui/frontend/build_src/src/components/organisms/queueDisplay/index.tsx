import React from "react";
import { ImageRequest } from "../../../api";

import { QueuedRequest, useRequestQueue } from "../../../stores/requestQueueStore";

import {
  QueueDisplayMain,
  QueueListButtons,
  CompleteButtton,
  ErrorButton
} from "./queueDisplay.css";

import {
  buttonStyle
} from "../../_recipes/button.css";

import ClearQueue from "../../molecules/clearQueue";
import QueueItem from "./queueItem";

export default function QueueDisplay() {

  const requests: QueuedRequest[] = useRequestQueue((state) => state.requests);
  const removeCompleted = useRequestQueue((state) => state.removeCompleted);
  const removeErrored = useRequestQueue((state) => state.removeErrored);

  const clearCompleted = () => {
    removeCompleted();
  }

  const clearErrored = () => {
    removeErrored();
  }

  return (
    <div className={QueueDisplayMain}>
      <ClearQueue />
      <div className={QueueListButtons}>
        <button
          className={buttonStyle({
            type: "secondary",
          })}
          onClick={clearCompleted}>Clear Completed</button>
        <button
          className={buttonStyle({
            type: "secondary",
          })}
          onClick={clearErrored}>Clear Errored</button>
      </div>
      {requests.map((request) => {
        return <QueueItem key={request.id} request={request}></QueueItem>;
      })}
    </div>
  );
}
