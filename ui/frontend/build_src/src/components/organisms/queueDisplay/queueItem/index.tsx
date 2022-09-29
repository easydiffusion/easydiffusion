import React from "react";

import {
  ImageRequest
} from '../../../../api'

import { QueueStatus, QueuedRequest, useRequestQueue } from '../../../../stores/requestQueueStore';

import StopButton from '../../../molecules/stopButton';

import {
  QueueItemMain,
  QueueButtons,
  CompleteButtton,
  PauseButton,
  ResumeButton,
  CancelButton,
  RetryButton,
  SendToTopButton,
} from "./queueItem.css";


interface QueueItemProps {
  request: QueuedRequest;
}

export default function QueueItem({ request }: QueueItemProps) {

  // console.log('info', info);
  // console.log('status', status);

  const removeItem = useRequestQueue((state) => state.removeItem);
  const updateStatus = useRequestQueue((state) => state.updateStatus);
  const sendPendingToTop = useRequestQueue((state) => state.sendPendingToTop);

  const {
    id,
    options: {
      prompt,
      seed,
      sampler,
    },
    status,
  } = request;

  const removeFromQueue = () => {
    console.log('remove from queue');
    removeItem(id);
  }

  const pauseItem = () => {
    console.log('pause item');
    updateStatus(id, QueueStatus.paused);
  }

  const retryRequest = () => {
    console.log('retry request');
    updateStatus(id, QueueStatus.pending);
  }

  const sendToTop = () => {
    console.log('send to top');
    sendPendingToTop(id);
  }

  return (
    <div className={[QueueItemMain, status].join(' ')}>
      {/* @ts-expect-error */}
      <div>{status}</div>
      <div>{prompt}</div>
      <div>{seed}</div>
      <div>{sampler}</div>

      <div className={QueueButtons}>

        {status === QueueStatus.processing && (
          <StopButton></StopButton>
        )}

        {status === QueueStatus.complete && (
          <button className={CompleteButtton} onClick={removeFromQueue}>Clear</button>
        )}

        {status === QueueStatus.pending && (
          <>
            <button className={CancelButton} onClick={removeFromQueue}>Remove</button>
            <button className={PauseButton} onClick={pauseItem}>Pause</button>
            <button className={SendToTopButton} onClick={sendToTop}>Send to top</button>
          </>
        )}

        {status === QueueStatus.paused && (
          <button className={ResumeButton} onClick={retryRequest}>Resume</button>
        )}

        {status === QueueStatus.error && (
          <button className={RetryButton} onClick={retryRequest}>Retry</button>
        )}
      </div>

    </div>
  );
}