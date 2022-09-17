// @ts-nocheck
import React, { useRef, useState } from "react";

// https://github.com/embiem/react-canvas-draw

type DrawImageProps = {
  imageData: string;
};

import {
  DrawImageMain, //@ts-ignore
} from "./drawImage.css.ts";

export default function DrawImage({ imageData }: DrawImageProps) {

  const drawingRef = useRef<HTMLCanvasElement>(null);
  const cursorRef = useRef<HTMLCanvasElement>(null);

  const [isDrawing, setIsDrawing] = useState(false);

  const _handleMouseDown = (
    e: React.MouseEvent<HTMLCanvasElement, MouseEvent>
  ) => {
    console.log("mouse down", e);

    const {
      nativeEvent: { offsetX, offsetY },
    } = e;

    setIsDrawing(true);
  };

  const _handleMouseUp = (
    e: React.MouseEvent<HTMLCanvasElement, MouseEvent>
  ) => {
    setIsDrawing(false);

    const canvas = drawingRef.current;
    if (canvas) {
      const data = canvas.toDataURL();
      console.log("data", data);
    }
  };

  const _handleMouseMove = (
    e: React.MouseEvent<HTMLCanvasElement, MouseEvent>
  ) => {
    if (isDrawing) {
      const canvas = drawingRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx.strokeStyle = "red";
        const {
          nativeEvent: { offsetX, offsetY },
        } = e;

        ctx.beginPath();

        ctx.lineWidth = 20;

        // Sets the end of the lines drawn
        // to a round shape.
        ctx.lineCap = "round";

        ctx.strokeStyle = "white";
        // The cursor to start drawing
        // moves to this coordinate
        ctx.moveTo(offsetX, offsetY);

        // A line is traced from start
        // coordinate to this coordinate
        ctx.lineTo(offsetX, offsetY);

        // Draws the line.
        ctx.stroke();
      }
    }
  };

  const _handleCursorMove = (
    e: React.MouseEvent<HTMLCanvasElement, MouseEvent>
  ) => {
    console.log("cursor move");


    const canvas = cursorRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.strokeStyle = "red";
      const {
        nativeEvent: { offsetX, offsetY },
      } = e;

      ctx.beginPath();


      ctx.clearRect(0, 0, canvas.width, canvas.height);

      ctx.lineWidth = 20;

      // Sets the end of the lines drawn
      // to a round shape.
      ctx.lineCap = "round";

      ctx.strokeStyle = "white";
      // The cursor to start drawing
      // moves to this coordinate
      ctx.moveTo(offsetX, offsetY);

      // A line is traced from start
      // coordinate to this coordinate
      ctx.lineTo(offsetX, offsetY);

      // Draws the line.
      ctx.stroke();
    }
  };


  return (
    <div className={DrawImageMain}>
      <img src={imageData} />
      <canvas
        ref={drawingRef}
        width={512}
        height={512}
        onMouseDown={_handleMouseDown}
        onMouseMove={_handleMouseMove}
        onMouseUp={_handleMouseUp}
      ></canvas>
      <canvas
        ref={cursorRef}
        width={512}
        height={512}
        onMouseMove={_handleCursorMove}
      ></canvas>

    </div>
  );
}
