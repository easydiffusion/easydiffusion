import React, { ChangeEvent } from "react";

import NumberRange from "@atoms/numberRange";
import { useImageCreate } from "@stores/imageCreateStore";


export default function GuidanceScale() {
  const setRequestOption = useImageCreate((state) => state.setRequestOptions);
  const guidanceScale = useImageCreate((state) =>
    state.getValueForRequestKey("guidance_scale")
  );

  const handleGuidanceScaleChange = (event: ChangeEvent<HTMLInputElement>) => {
    const valu = event.target.value
    setRequestOption("guidance_scale", valu);
  };


  return (
    <div>
      <NumberRange
        min={0}
        max={50}
        step={1}
        value={guidanceScale}
        onChange={handleGuidanceScaleChange}
        shouldShowValue={true}
      />
    </div>
  );
}