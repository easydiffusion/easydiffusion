import React from "react";
import DrawImage from "../../../molecules/drawImage";

import { useImageCreate } from "../../../../stores/imageCreateStore";

export default function InpaintingPanel() {


  const init_image = useImageCreate((state) =>
    state.getValueForRequestKey("init_image")
  );


  return (
    <div>
      <DrawImage imageData={init_image} />
    </div>
  );
};
