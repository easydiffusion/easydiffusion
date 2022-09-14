import React from "react";

import StatusDisplay from "./statusDisplay";

import './headerDisplay.css';

export default function HeaderDisplay() {
  return (
    <div className="header-display">
      <h1>Stable Diffusion UI v2.1.0</h1>
      <StatusDisplay className="status-display"></StatusDisplay>
    </div>
  );
};