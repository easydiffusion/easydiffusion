/* eslint-disable  @typescript-eslint/naming-convention */

import React from "react";

import {
  ImageRequest
} from '../../../../api'

import { QueueStatus, QueuedRequest, useRequestQueue } from '../../../../stores/requestQueueStore';

import StopButton from '../../../molecules/stopButton';

import {
  QueueItemMain,
  QueueItemInfo,
  QueueButtons,
  // CompleteButtton,
  // PauseButton,
  // ResumeButton,
  // CancelButton,
  // RetryButton,
  // SendToTopButton,
} from "./queueItem.css";


import {
  buttonStyle
} from "../../../_recipes/button.css";


interface QueueItemProps {
  request: QueuedRequest;
}

export default function QueueItem({ request }: QueueItemProps) {

  const removeItem = useRequestQueue((state) => state.removeItem);
  const updateStatus = useRequestQueue((state) => state.updateStatus);
  const sendPendingToTop = useRequestQueue((state) => state.sendPendingToTop);

  const {
    id,
    options: {
      prompt,
      num_outputs,
      seed,
      sampler,
      guidance_scale,
      num_inference_steps,

    },
    status,
  } = request;

  const removeFromQueue = () => {
    removeItem(id);
  }

  const pauseItem = () => {
    updateStatus(id, QueueStatus.paused);
  }

  const retryRequest = () => {
    updateStatus(id, QueueStatus.pending);
  }

  const sendToTop = () => {
    sendPendingToTop(id);
  }

  return (
    <div className={[QueueItemMain, status].join(' ')}>

      <div className={QueueItemInfo}>
        <p>{prompt}</p>
        <p>Making {num_outputs} concurrent images</p>
        <p>
          <span>Seed: {seed} </span>
          <span>Sampler: {sampler} </span>
          <span>Guidance Scale: {guidance_scale} </span>
          <span>Num Inference Steps: {num_inference_steps} </span>
        </p>
      </div>

      <div className={QueueButtons}>

        {status === QueueStatus.processing && (
          <StopButton></StopButton>
        )}

        {status === QueueStatus.complete && (
          <button
            className={buttonStyle({
              size: "large",
            })}
            onClick={removeFromQueue}>
            Clear
          </button>
        )}

        {status === QueueStatus.pending && (
          <>
            <button className={buttonStyle({
              type: "cancel",
            })} onClick={removeFromQueue}>Remove</button>
            <button className={buttonStyle({
              type: "secondary",
            })} onClick={pauseItem}>Pause</button>
            <button className={buttonStyle({
              type: "secondary",
            })} onClick={sendToTop}>Send to top</button>
          </>
        )}

        {status === QueueStatus.paused && (
          <button
            className={buttonStyle({
              size: "large",
            })} onClick={retryRequest}>Resume</button>
        )}

        {status === QueueStatus.error && (
          <button
            className={buttonStyle({
              size: "large",
            })} onClick={retryRequest}>Retry</button>
        )}
      </div>

    </div>
  );
}