/* eslint-disable @typescript-eslint/restrict-template-expressions */
/* eslint-disable multiline-ternary */
/* eslint-disable @typescript-eslint/naming-convention */
import React, { useEffect, useState } from "react";
import GeneratedImage from "../../../molecules/generatedImage";
import { ImageRequest, useImageCreate } from "../../../../stores/imageCreateStore";
import { FetchingStates, useImageFetching } from "../../../../stores/imageFetchingStore";
import { useImageDisplay } from "../../../../stores/imageDisplayStore";


export interface CompletedImagesType {
  id?: string;
  data: string | undefined;
  info: ImageRequest | undefined;
}

const IdleDisplay = () => {
  return (
    <h4 className="no-image">Try Making a new image!</h4>
  );
};

const LoadingDisplay = () => {

  const step = useImageFetching((state) => state.step);
  const totalSteps = useImageFetching((state) => state.totalSteps);
  const progressImages = useImageFetching((state) => state.progressImages);

  const [percent, setPercent] = useState(0);

  console.log("progressImages", progressImages);



  useEffect(() => {
    if (totalSteps > 0) {
      setPercent(Math.round((step / totalSteps) * 100));
    } else {
      setPercent(0);
    }
  }, [step, totalSteps]);

  return (
    <>
      <h4 className="loading">Loading...</h4>
      <p>{percent} % Complete </p>
      {progressImages.map((image, index) => {
        return (
          <img src={`http://localhost:9000${image}`} key={index} />
        )
      })
      }
    </>
  );
};

const ImageDisplay = ({ info, data }: CompletedImagesType) => {

  const createFileName = () => {
    const {
      prompt,
      seed,
      num_inference_steps,
      guidance_scale,
      use_face_correction,
      use_upscale,
      width,
      height,
    } = info;

    // Most important information is the prompt
    let underscoreName = prompt.replace(/[^a-zA-Z0-9]/g, "_");
    underscoreName = underscoreName.substring(0, 100);
    // name and the top level metadata
    let fileName = `${underscoreName}_Seed-${seed}_Steps-${num_inference_steps}_Guidance-${guidance_scale}`;
    // Add the face correction and upscale
    if (typeof use_face_correction == "string") {
      fileName += `_FaceCorrection-${use_face_correction}`;
    }
    if (typeof use_upscale == "string") {
      fileName += `_Upscale-${use_upscale}`;
    }
    // Add the width and height
    fileName += `_${width}x${height}`;
    // add the file extension
    fileName += ".png";
    // return fileName
    return fileName;
  };



  const setRequestOption = useImageCreate((state) => state.setRequestOptions);

  const _handleSave = () => {
    const link = document.createElement("a");
    link.download = createFileName();
    link.href = data!;
    link.click();
  };

  const _handleUseAsInput = () => {
    setRequestOption("init_image", data);
  };



  return (
    <div className="imageDisplay">
      <p> {info?.prompt}</p>
      <GeneratedImage imageData={data} metadata={info}></GeneratedImage>
      <div>
        <button onClick={_handleSave}>Save</button>
        <button onClick={_handleUseAsInput}>Use as Input</button>
      </div>
    </div>
  );
};

export default function CurrentDisplay() {

  const status = useImageFetching((state) => state.status);

  const [currentImage, setCurrentImage] = useState<CompletedImagesType | null>(
    null
  );
  // const testCur = useImageDisplay((state) => state.getCurrentImage());
  const images = useImageDisplay((state) => state.images);

  useEffect(() => {
    if (images.length > 0) {
      console.log("cur", images[0]);
      setCurrentImage(images[0]);
    } else {
      setCurrentImage(null);
    }
  }, [images]);

  return (
    <div className="current-display">

      {status === FetchingStates.IDLE && <IdleDisplay />}

      {(status === FetchingStates.FETCHING || status === FetchingStates.PROGRESSING) && <LoadingDisplay />}

      {(status === FetchingStates.COMPLETE && currentImage != null) && <ImageDisplay info={currentImage?.info} data={currentImage?.data} />}

    </div>
  );
}