import React, { useRef, useState, useEffect } from "react";

import {
  DrawImageMain,
} from "./drawImage.css";

// https://github.com/embiem/react-canvas-draw

interface DrawImageProps {
  imageData: string;
  brushSize: number;

  brushShape: string;
  brushColor: string;
  isErasing: boolean;
  setData: (data: string) => void;
}

export default function DrawImage({
  imageData,
  brushSize,
  brushShape,
  brushColor,
  isErasing,
  setData,
}: DrawImageProps) {
  const drawingRef = useRef<HTMLCanvasElement>(null);
  const cursorRef = useRef<HTMLCanvasElement>(null);
  const [isUpdating, setIsUpdating] = useState(false);

  const [canvasWidth, setCanvasWidth] = useState(512);
  const [canvasHeight, setCanvasHeight] = useState(512);

  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      setCanvasWidth(img.width);
      setCanvasHeight(img.height);
    };
    img.src = imageData;
  }, [imageData]);

  useEffect(() => {
    // when the brush color changes, change the color of all the
    // drawn pixels to the new color
    if (drawingRef.current != null) {
      const ctx = drawingRef.current.getContext("2d");
      if (ctx != null) {

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
    }
  }, [brushColor]);

  const _handleMouseDown = (
    e: React.MouseEvent<HTMLCanvasElement, MouseEvent>
  ) => {
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
    if (canvas != null) {
      const data = canvas.toDataURL();
      setData(data);
    }
  };

  const _drawCanvas = (x: number, y: number, brushSize: number, brushShape: string, brushColor: string | CanvasGradient | CanvasPattern) => {
    const canvas = drawingRef.current;
    if (canvas != null) {
      const ctx = canvas.getContext("2d");
      if (ctx != null) {
        if (isErasing) {
          // stack overflow https://stackoverflow.com/questions/10396991/clearing-circular-regions-from-html5-canvas
          const offset = brushSize / 2;
          ctx.clearRect(x - offset, y - offset, brushSize, brushSize);
        } else {
          ctx.beginPath();
          ctx.lineWidth = brushSize;
          // @ts-expect-error
          ctx.lineCap = brushShape;
          ctx.strokeStyle = brushColor;
          ctx.moveTo(x, y);
          ctx.lineTo(x, y);
          ctx.stroke();
        }
      }
    }
  };

  const _drawCursor = (
    x: number,
    y: number,
    brushSize: number,
    brushShape: string,
    brushColor: string
  ) => {
    const canvas = cursorRef.current;
    if (canvas != null) {
      const ctx = canvas.getContext("2d");
      if (ctx != null) {
        ctx.beginPath();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (isErasing) {
          const offset = brushSize / 2;
          // draw a quare
          ctx.lineWidth = 2;
          ctx.lineCap = "butt";
          ctx.strokeStyle = brushColor;
          ctx.moveTo(x - offset, y - offset);
          ctx.lineTo(x + offset, y - offset);
          ctx.lineTo(x + offset, y + offset);
          ctx.lineTo(x - offset, y + offset);
          ctx.lineTo(x - offset, y - offset);
          ctx.stroke();
        } else {
          ctx.lineWidth = brushSize;
          // @ts-expect-error
          ctx.lineCap = brushShape;
          ctx.strokeStyle = brushColor;
          ctx.moveTo(x, y);
          ctx.lineTo(x, y);
          ctx.stroke();
        }
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
    if (canvas != null) {
      const ctx = canvas.getContext("2d");
      if (ctx != null) {
        ctx.fillStyle = brushColor;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }
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
