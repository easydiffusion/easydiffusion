import React, {useCallback} from "react";


import { ImageRequest, useImageCreate } from "../../../store/imageCreateStore";

type GeneretaedImageProps = {
  imageData: string;
  metadata: ImageRequest;
}


export default function GeneratedImage({ imageData, metadata}: GeneretaedImageProps) {

  const setRequestOption = useImageCreate((state) => state.setRequestOptions);
  
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
    } = metadata;

    //Most important information is the prompt
    let underscoreName = prompt.replace(/[^a-zA-Z0-9]/g, '_')
    underscoreName = underscoreName.substring(0, 100)

    // name and the top level metadata
    let fileName = `${underscoreName}_Seed-${seed}_Steps-${num_inference_steps}_Guidance-${guidance_scale}`;

    // Add the face correction and upscale
    if(use_face_correction) {
      fileName += `_FaceCorrection-${use_face_correction}`;
    }
    if(use_upscale) {
      fileName += `_Upscale-${use_upscale}`;
    }

    // Add the width and height
    fileName += `_${width}x${height}`;

    // add the file extension
    fileName += `.png`

    // return fileName
    return fileName;
  }
  
  const _handleSave = () => {
    const link = document.createElement("a");
    link.download = createFileName();
    link.href = imageData;
    link.click();
  };

  const _handleUseAsInput = () => {
    console.log(" TODO : use as input");


    setRequestOption("init_image", imageData);
    // initImageSelector.value = null
    // initImagePreview.src = imgBody


    // imgUseBtn.addEventListener('click', function() {
    //   initImageSelector.value = null
    //   initImagePreview.src = imgBody

    //   initImagePreviewContainer.style.display = 'block'
    //   promptStrengthContainer.style.display = 'block'

    //   // maskSetting.style.display = 'block'

    //   randomSeedField.checked = false
    //   seedField.value = seed
    //   seedField.disabled = false
    // })
  }

  return (
    <div className="generated-image">
      <p>Your image</p>
      <img src={imageData} alt="generated" />
      <button onClick={_handleSave}>
        Save
      </button>
      <button onClick={_handleUseAsInput}>
        Use as Input
      </button>
    </div>
  );
}