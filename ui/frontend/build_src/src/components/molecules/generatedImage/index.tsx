import React from "react";

import { ImageRequest } from "../../../api";

import {
  generatedImageMain,
  image,
} from "./generatedImage.css";

interface GeneretaedImageProps {
  imageData: string | undefined;
  metadata: ImageRequest | undefined;
  className?: string;
  // children: never[];
}

export default function GeneratedImage({
  imageData,
  metadata,
  className,
}: GeneretaedImageProps) {
  return (
    <div className={[generatedImageMain, className].join(" ")}>
      <img className={image} src={imageData} alt={metadata!.prompt} />
    </div>
  );
}
