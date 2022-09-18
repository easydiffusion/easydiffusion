import React, { useCallback } from "react";

import { ImageRequest, useImageCreate } from "../../../stores/imageCreateStore";

import {
  generatedImageMain,
  image, //@ts-ignore
} from "./generatedImage.css.ts";

type GeneretaedImageProps = {
  imageData: string;
  metadata: ImageRequest;
  className?: string;
  // children: never[];
};

export default function GeneratedImage({
  imageData,
  metadata,
  className,
}: GeneretaedImageProps) {
  return (
    <div className={[generatedImageMain, className].join(" ")}>
      <img className={image} src={imageData} alt={metadata.prompt} />
    </div>
  );
}
