import React, { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { healthPing, HEALTH_PING_INTERVAL } from "../../../api";

const startingMessage = "Stable Diffusion is starting...";
const successMessage = "Stable Diffusion is ready to use!";
const errorMessage = "Stable Diffusion is not running!";

import "./statusDisplay.css";

export default function StatusDisplay({ className }: { className?: string }) {
  const [statusMessage, setStatusMessage] = useState(startingMessage);
  const [statusClass, setStatusClass] = useState("starting");

  // but this will be moved to the status display when it is created
  const { status, data } = useQuery(["health"], healthPing, {
    refetchInterval: HEALTH_PING_INTERVAL,
  });

  useEffect(() => {
    if (status === "loading") {
      setStatusMessage(startingMessage);
      setStatusClass("starting");
    } else if (status === "error") {
      setStatusMessage(errorMessage);
      setStatusClass("error");
    } else if (status === "success") {
      if (data[0] === "OK") {
        setStatusMessage(successMessage);
        setStatusClass("success");
      } else {
        setStatusMessage(errorMessage);
        setStatusClass("error");
      }
    }
  }, [status, data]);

  return (
    <>
      {/* alittle hacky but joins the class names, will probably need a better css in js solution or tailwinds*/}
      <p className={[statusClass, className].join(" ")}>{statusMessage}</p>
    </>
  );
}
