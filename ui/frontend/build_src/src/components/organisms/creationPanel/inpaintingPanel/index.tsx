import React, { useRef, useState, ChangeEvent, MouseEventHandler } from "react";
import DrawImage from "../../../molecules/drawImage";

import { useImageCreate } from "../../../../stores/imageCreateStore";

import {
  InpaintingPanelMain,
  InpaintingControls,
  InpaintingControlRow, // @ts-expect-error
} from "./inpaintingPanel.css.ts";

export default function InpaintingPanel() {
  // no idea if this is the right typing
  const drawingRef = useRef(null);

  const [brushSize, setBrushSize] = useState("20");
  const [brushShape, setBrushShape] = useState("round");
  const [brushColor, setBrushColor] = useState("#fff");
  const [isErasing, setIsErasing] = useState(false);

  const init_image = useImageCreate((state) =>
    state.getValueForRequestKey("init_image")
  );

  const _handleBrushMask = () => {
    setIsErasing(false);
  };

  const _handleBrushErase = () => {
    setIsErasing(true);
  };

  const _handleFillMask = () => {
    console.log("fill mask!!", drawingRef);

    // drawingRef.current?.fillCanvas();
  };

  const _handleClearAll = () => {
    console.log("clear all");
  };

  const _handleBrushSize = (event: ChangeEvent<HTMLInputElement>) => {
    setBrushSize(event.target.value);
  };

  return (
    <div className={InpaintingPanelMain}>
      <DrawImage
        // ref={drawingRef}
        imageData={init_image}
        brushSize={brushSize}
        brushShape={brushShape}
        brushColor={brushColor}
        isErasing={isErasing}
      />
      <div className={InpaintingControls}>
        <div className={InpaintingControlRow}>
          <button onClick={_handleBrushMask}>Mask</button>
          <button onClick={_handleBrushErase}>Erase</button>
          <button disabled onClick={_handleFillMask}>
            Fill
          </button>
          <button disabled onClick={_handleClearAll}>
            Clear
          </button>

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

          <button
            onClick={() => {
              setBrushColor("#000");
            }}
          >
            Dark Brush
          </button>
          <button
            onClick={() => {
              setBrushColor("#fff");
            }}
          >
            Light Brush
          </button>
        </div>
      </div>
    </div>
  );
}
