import React, { useCallback } from "react";

import { ImageRequest, useImageCreate } from "../../../stores/imageCreateStore";

import {
  generatedImageMain,
  image, // @ts-expect-error
} from "./generatedImage.css.ts";

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
