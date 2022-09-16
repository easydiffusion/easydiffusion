// @ts-nocheck
import React, { useRef, useEffect } from "react";

// https://github.com/embiem/react-canvas-draw

type DrawImageProps = {
  imageData: string;
};

import {
  DrawImageMain
} from //@ts-ignore
  './drawImage.css.ts';

export default function DrawImage({ imageData }: DrawImageProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const draw = (ctx: CanvasRenderingContext2D) => {
    ctx.fillStyle = "red";
    ctx.fillRect(0, 0, 100, 100);
  };

  // useEffect(() => {
  //   const canvas = canvasRef.current;
  //   if (canvas) {
  //     if (imageData) {
  //       const ctx = canvas.getContext("2d");
  //       if (ctx) {
  //         const img = new Image();
  //         img.onload = () => {
  //           ctx.drawImage(img, 0, 0);
  //         };
  //         img.src = imageData;
  //       }
  //     } else {
  //       const ctx = canvas.getContext("2d");
  //       if (ctx) {
  //         draw(ctx);
  //       }
  //     }
  //   } else {
  //     console.log("canvas is null");
  //   }
  // }, [imageData, draw]);

  const _handleMouseDown = (
    e: React.MouseEvent<HTMLCanvasElement, MouseEvent>
  ) => {
    console.log("mouse down", e);
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.strokeStyle = "#red";
      const {
        nativeEvent: { offsetX, offsetY },
      } = e;

      // console.log("x: " + x + " y: " + y);
      ctx.beginPath();
      ctx.moveTo(offsetX, offsetY);
      // ctx.lineTo(x + 1, y + 1);
      // ctx.stroke();
    }
  };

  const _handleMouseUp = (
    e: React.MouseEvent<HTMLCanvasElement, MouseEvent>
  ) => {
    console.log("mouse up");
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) {
        // draw(ctx);
      }
      const {
        nativeEvent: { offsetX, offsetY },
      } = e;

      // console.log("x: " + x + " y: " + y);

      // ctx.moveTo(x, y);
      ctx?.lineTo(offsetX, offsetY);
      ctx.stroke();
      ctx.closePath();
    }
  };

  return (
    <div className={DrawImageMain}>
      <img src={imageData} />
      <canvas
        ref={canvasRef}
        width={512}
        height={512}
        onMouseDown={_handleMouseDown}
        onMouseUp={_handleMouseUp}
      ></canvas>
    </div>
  );
}
