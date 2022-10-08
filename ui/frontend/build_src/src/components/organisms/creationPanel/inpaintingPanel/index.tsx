import React, { useRef, useState, ChangeEvent } from "react";
import DrawImage from "../../../molecules/_stateless/drawImage";

import { useImageCreate } from "../../../../stores/imageCreateStore";

import {
  InpaintingPanelMain,
  InpaintingControls,
  InpaintingControlRow,
} from "./inpaintingPanel.css";

export default function InpaintingPanel() {
  // no idea if this is the right typing
  // const drawingRef = useRef(null);

  const [brushSize, setBrushSize] = useState("20");
  const [brushShape, setBrushShape] = useState("round");
  const [brushColor, setBrushColor] = useState("#fff");
  const [isErasing, setIsErasing] = useState(false);

  const initImage = useImageCreate((state) =>
    state.getValueForRequestKey("init_image")
  );

  const setRequestOption = useImageCreate((state) => state.setRequestOptions);

  const setMask = (data: string) => {
    setRequestOption("mask", data);
  }


  const _handleBrushMask = () => {
    setIsErasing(false);
  };

  const _handleBrushErase = () => {
    setIsErasing(true);
  };


  const _handleBrushSize = (event: ChangeEvent<HTMLInputElement>) => {
    setBrushSize(event.target.value);
  };

  return (
    <div className={InpaintingPanelMain}>
      <DrawImage
        // ref={drawingRef}
        imageData={initImage}
        // @ts-expect-error
        brushSize={brushSize}
        brushShape={brushShape}
        brushColor={brushColor}
        isErasing={isErasing}
        setData={setMask}
      />
      <div className={InpaintingControls}>
        <div className={InpaintingControlRow}>
          <button onClick={_handleBrushMask}>Mask</button>
          <button onClick={_handleBrushErase}>Erase</button>
          {/* <button disabled onClick={_handleFillMask}>
            Fill
          </button>
          <button disabled onClick={_handleClearAll}>
            Clear
          </button> */}

          <label>
            Brush Size
            <input
              type="range"
              min="1"
              max="100"
              value={brushSize}
              onChange={_handleBrushSize}
            ></input>
          </label>
        </div>

        <div className={InpaintingControlRow}>
          <button
            onClick={() => {
              setBrushShape("round");
            }}
          >
            Cirle Brush
          </button>
          <button
            onClick={() => {
              setBrushShape("square");
            }}
          >
            Square Brush
          </button>

          {/* <button
            onClick={() => {
              setBrushColor("#000");
            }}
          >
            Dark Brush
          </button> */}
          {/* <button
            onClick={() => {
              setBrushColor("#fff");
            }}
          >
            Light Brush
          </button> */}
        </div>
      </div>
    </div>
  );
}
