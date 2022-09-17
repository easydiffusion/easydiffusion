// @ts-nocheck
import React, { useRef, useState, useEffect } from "react";

// https://github.com/embiem/react-canvas-draw

type DrawImageProps = {
  imageData: string;
  brushSize: string;

  brushShape: string;
  brushColor: string;
  isErasing: boolean;

};

import {
  DrawImageMain, //@ts-ignore
} from "./drawImage.css.ts";

export default function DrawImage({ imageData, brushSize, brushShape, brushColor, isErasing }: DrawImageProps) {

  const drawingRef = useRef<HTMLCanvasElement>(null);
  const cursorRef = useRef<HTMLCanvasElement>(null);
  const [isUpdating, setIsUpdating] = useState(false);

  const [canvasWidth, setCanvasWidth] = useState(512);
  const [canvasHeight, setCanvasHeight] = useState(512);


  useEffect(() => {
    console.log(imageData);
    const img = new Image();
    img.onload = () => {
      setCanvasWidth(img.width);
      setCanvasHeight(img.height);

    }
    img.src = imageData;

  }, [imageData]);


  useEffect(() => {
    // when the brush color changes, change the color of all the 
    // drawn pixels to the new color
    if (drawingRef.current) {
      const ctx = drawingRef.current.getContext("2d");
      const imageData = ctx.getImageData(0, 0, canvasWidth, canvasHeight);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        if (data[i + 3] > 0) {
          data[i] = parseInt(brushColor, 16);
          data[i + 1] = parseInt(brushColor, 16);
          data[i + 2] = parseInt(brushColor, 16);
        }
      }
      ctx.putImageData(imageData, 0, 0);
    }

  }, [brushColor]);

  const _handleMouseDown = (
    e: React.MouseEvent<HTMLCanvasElement, MouseEvent>
  ) => {
    console.log("mouse down", e);

    const {
      nativeEvent: { offsetX, offsetY },
    } = e;

    setIsUpdating(true);
  };

  const _handleMouseUp = (
    e: React.MouseEvent<HTMLCanvasElement, MouseEvent>
  ) => {
    setIsUpdating(false);
    const canvas = drawingRef.current;
    if (canvas) {
      const data = canvas.toDataURL();
      // TODO: SEND THIS TO THE STATE
    }
  };

  const _drawCanvas = (x, y, brushSize, brushShape, brushColor) => {
    const canvas = drawingRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (isErasing) {

        // stack overflow https://stackoverflow.com/questions/10396991/clearing-circular-regions-from-html5-canvas

        const offset = brushSize / 2;
        ctx.clearRect(x - offset, y - offset, brushSize, brushSize);

      } else {
        ctx.beginPath();
        ctx.lineWidth = brushSize;
        ctx.lineCap = brushShape;
        ctx.strokeStyle = brushColor;
        ctx.moveTo(x, y);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
    }
  };

  const _drawCursor = (x, y, brushSize, brushShape, brushColor) => {
    const canvas = cursorRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.beginPath();
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (isErasing) {
        const offset = brushSize / 2;
        // draw a quare
        ctx.lineWidth = 2;
        ctx.lineCap = 'butt';
        ctx.strokeStyle = brushColor;
        ctx.moveTo(x - offset, y - offset);
        ctx.lineTo(x + offset, y - offset);
        ctx.lineTo(x + offset, y + offset);
        ctx.lineTo(x - offset, y + offset);
        ctx.lineTo(x - offset, y - offset);
        ctx.stroke();

      } else {

        ctx.lineWidth = brushSize;
        ctx.lineCap = brushShape;
        ctx.strokeStyle = brushColor;
        ctx.moveTo(x, y);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
    }

  };

  const _handleMouseMove = (
    e: React.MouseEvent<HTMLCanvasElement, MouseEvent>
  ) => {

    const {
      nativeEvent: { offsetX: x, offsetY: y },
    } = e;

    _drawCursor(x, y, brushSize, brushShape, brushColor);

    if (isUpdating) {
      _drawCanvas(x, y, brushSize, brushShape, brushColor);
    }
  };

  // function for external use
  const fillCanvas = () => {
    const canvas = drawingRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = brushColor;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
  };

  return (
    <div className={DrawImageMain}>
      <img src={imageData} />
      <canvas
        ref={drawingRef}
        width={canvasWidth}
        height={canvasHeight}
      ></canvas>
      <canvas
        ref={cursorRef}
        width={canvasWidth}
        height={canvasHeight}
        onMouseDown={_handleMouseDown}
        onMouseUp={_handleMouseUp}
        onMouseMove={_handleMouseMove}
      ></canvas>
    </div>
  );
}
