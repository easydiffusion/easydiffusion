import React, {useRef, useEffect} from "react";

// https://github.com/embiem/react-canvas-draw

type DrawImageProps = {
  imageData: string;
};


export default function DrawImage({imageData}: DrawImageProps) {

  const canvasRef = useRef<HTMLCanvasElement>(null);

  const draw = (ctx: CanvasRenderingContext2D) => {
    ctx.fillStyle = "red";
    ctx.fillRect(0, 0, 100, 100);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
    
      if(imageData){
        const ctx = canvas.getContext("2d");
        if (ctx) {
          const img = new Image();
          img.onload = () => {
            ctx.drawImage(img, 0, 0);
          };
          img.src = imageData;
        }
      }
      else {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          draw(ctx);
        }
      }
    }
    else {
      console.log("canvas is null");
    }
  }, [imageData, draw]);


  const _handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement, MouseEvent>) => {
    console.log("mouse down", e);
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.strokeStyle = '#ff0000';
      const {nativeEvent: {x, y}} = e;

      console.log("x: " + x + " y: " + y);

      ctx.moveTo(x,y);
      ctx.lineTo(x+1,y+1);
      ctx.stroke();
    }
  };

  const _handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement, MouseEvent>) => {
    console.log("mouse up");
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      // if (ctx) {
      //   draw(ctx);
      // }
      const {nativeEvent: {x, y}} = e;

      ctx.moveTo(x,y);
      ctx.lineTo(x+1,y+1);
      ctx.stroke();
      ctx.closePath();
    }
  };


  return (
    <div>
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