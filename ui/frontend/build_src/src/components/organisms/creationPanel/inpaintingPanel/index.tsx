import React, { useRef, useState, ChangeEvent } from "react";
import DrawImage from "../../../molecules/drawImage";

import { useImageCreate } from "../../../../stores/imageCreateStore";


import {
  InpaintingPanelMain,
  InpaintingControls,
  InpaintingControlRow,
} from // @ts-ignore
  "./inpaintingPanel.css.ts";

export default function InpaintingPanel() {

  // no idea if this is the right typing
  const drawingRef = useRef(null);

  const [brushSize, setBrushSize] = useState('20');
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

  const _handleBrushSize = (e: ChangeEvent<HTMLInputElement>) => {
    setBrushSize(e.target.value);
  };

  // const _handleBrushShape = (e: ) => {
  //   console.log("brush shape", e.target.value);
  //   setBrushShape(e.target.value);
  // };

  const _handleBrushShape = (e: ChangeEvent<HTMLInputElement>) => {
    console.log("brush shape", e.target.value);
    setBrushShape(e.target.value);
  };

  const _handleBrushColor = (e: ChangeEvent<HTMLInputElement>) => {
    console.log("brush color", e.target.value);
    setBrushColor(e.target.value);
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
          <button
            onClick={_handleBrushMask}
          >
            Mask
          </button>
          <button
            onClick={_handleBrushErase}
          >
            Erase
          </button>
          <button
            disabled
            onClick={_handleFillMask}
          >
            Fill
          </button>
          <button
            disabled
            onClick={_handleClearAll}
          >
            Clear
          </button>

          <label
          >
            Brush Size
            <input
              type="range"
              min="1"
              max="100"
              value={brushSize}
              onChange={_handleBrushSize}
            >
            </input>
          </label>
        </div>

        <div className={InpaintingControlRow}>
          <button
            value={"round"}
            onClick={_handleBrushShape}
          >
            Cirle Brush
          </button>
          <button
            value={"square"}
            onClick={_handleBrushShape}
          >
            Square Brush
          </button>

          <button
            value={"#000"}
            onClick={_handleBrushColor}
          >
            Dark Brush
          </button>
          <button
            value={"#fff"}
            onClick={_handleBrushColor}
          >
            Light Brush
          </button>


        </div>

      </div>
    </div>
  );
};
