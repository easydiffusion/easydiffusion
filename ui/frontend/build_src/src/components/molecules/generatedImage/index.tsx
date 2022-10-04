import React from "react";

import { ImageRequest } from "../../../api/api.d";

import {
  generatedImageMain,
} from "./generatedImage.css";

interface GeneretaedImageProps {
  imageData: string | undefined;
  metadata: ImageRequest | undefined;
  className?: string;
}

export default function GeneratedImage({
  imageData,
  metadata,
  className,
}: GeneretaedImageProps) {
  return (
    <div className={[generatedImageMain, className].join(" ")}>
      <img src={imageData} alt={metadata!.prompt} />
    </div>
  );
}
