import React, { useEffect, useState, useRef } from "react";
import { useQuery } from "@tanstack/react-query";

import { healthPing, HEALTH_PING_INTERVAL } from "../../../../api";

import AudioDing from "../../../molecules/audioDing";

import {
  StartingStatus,
  ErrorStatus,
  SuccessStatus,
} from "./statusDisplay.css";

const startingMessage = "Stable Diffusion is starting...";
const successMessage = "Stable Diffusion is ready to use!";
const errorMessage = "Stable Diffusion is not running!";

export default function StatusDisplay({ className }: { className?: string }) {
  const [statusMessage, setStatusMessage] = useState(startingMessage);
  const [statusClass, setStatusClass] = useState(StartingStatus);

  const dingRef = useRef<HTMLAudioElement>(null);

  // but this will be moved to the status display when it is created
  const { status, data } = useQuery(["health"], healthPing, {
    refetchInterval: HEALTH_PING_INTERVAL,
  });

  useEffect(() => {
    if (status === "loading") {
      setStatusMessage(startingMessage);
      setStatusClass(StartingStatus);
    } else if (status === "error") {
      setStatusMessage(errorMessage);
      setStatusClass(ErrorStatus);
    } else if (status === "success") {
      if (data[0] === "OK") {
        setStatusMessage(successMessage);
        setStatusClass(SuccessStatus);
        // catch an auto play error
        dingRef.current?.play().catch((e) => {
          console.log('DING!')
        });
      } else {
        setStatusMessage(errorMessage);
        setStatusClass(ErrorStatus);
      }
    }
  }, [status, data, dingRef]);

  return (
    <>
      <AudioDing ref={dingRef}></AudioDing>
      <p className={[statusClass, className].join(" ")}>{statusMessage}</p>
    </>
  );
}
