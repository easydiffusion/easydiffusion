import React from "react";

import {
  QueueItemMain
} from "./queueItem.css";

import {
  ImageRequest
} from '../../../../api'

interface QueueItemProps {
  info: ImageRequest
}

export default function QueueItem({ info }: QueueItemProps) {

  console.log('info', info);

  const {
    id,
    options: {
      prompt,
      seed,
      sampler,
    },
    // status,
  } = info;

  return (
    <div className={QueueItemMain}>
      <div>{id}</div>
      <div>{prompt}</div>
      <div>{seed}</div>
      <div>{sampler}</div>

    </div>
  );
}