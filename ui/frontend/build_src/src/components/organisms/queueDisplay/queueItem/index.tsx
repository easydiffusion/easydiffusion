/* eslint-disable  @typescript-eslint/naming-convention */
import React from "react";

import { QueueStatus, QueuedRequest, useRequestQueue } from '../../../../stores/requestQueueStore';

import StopButton from '../../../molecules/stopButton';
import ProgressImageDisplay from "../../../molecules/progressImageDisplay";
import TimeRemaining from '../../../atoms/timeRemaining';

import {
  QueueItemMain,
  QueueItemInfo,
  QueueButtons,
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
    batchId,
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
    removeItem(batchId);
  }

  const pauseItem = () => {
    updateStatus(batchId, QueueStatus.paused);
  }

  const retryRequest = () => {
    updateStatus(batchId, QueueStatus.pending);
  }

  const reusePrompt = () => {
  }

  const sendToTop = () => {
    sendPendingToTop(batchId);
  }

  return (
    <div className={[QueueItemMain, status].join(' ')}>

      <div className={QueueItemInfo}>
        <p>{prompt}</p>
        {status === QueueStatus.processing && (
          <p>
            <span>Making {num_outputs} concurrent images </span>
            <span>Time: <TimeRemaining /></span>
          </p>
        )}
        <p>
          <span>Seed: {seed} </span>
          <span>Sampler: {sampler} </span>
          <span>Guidance Scale: {guidance_scale} </span>
          <span>Num Inference Steps: {num_inference_steps} </span>
        </p>
        <ProgressImageDisplay batchId={batchId}></ProgressImageDisplay>
      </div>

      <div className={QueueButtons}>

        {status === QueueStatus.processing && (
          <StopButton></StopButton>
        )}

        {status === QueueStatus.complete && (
          <>
            <button
              className={buttonStyle({
              })}
              onClick={removeFromQueue}>
              Clear
            </button>
            <button
              className={buttonStyle({
                color: "secondary",
              })}
              onClick={reusePrompt}>
              Reuse Prompt
            </button>
          </>
        )}

        {status === QueueStatus.pending && (
          <>
            <button className={buttonStyle({
              color: "cancel",
            })} onClick={removeFromQueue}>Remove</button>
            <button className={buttonStyle({
              color: "secondary",
              type: "outline",
            })} onClick={pauseItem}>Pause</button>
            <button className={buttonStyle({
              color: "tertiary",
              type: "action",
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