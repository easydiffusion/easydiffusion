import React from "react";

import { useImageQueue } from "../../../stores/imageQueueStore";


import {
  QueueDisplayMain
} from "./queueDisplay.css";

import QueueItem from "./queueItem";

export default function QueueDisplay() {

  const images = useImageQueue((state) => state.images);
  console.log('images', images);


  return (
    <div className={QueueDisplayMain}>
      {images.map((image) => {
        return <QueueItem key={image.id} info={image}></QueueItem>;
      })}
    </div>
  );
}
