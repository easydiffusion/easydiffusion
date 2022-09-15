import React, { ChangeEvent } from "react";

import MakeButton from "./basicCreation/makeButton";
import AdvancedSettings from "./advancedSettings";
import ImageModifiers from "./imageModifiers";

import "./creationPanel.css";

// @ts-ignore
import { CreationPaneMain } from "./creationpane.css.ts";

import BasicCreation from "./basicCreation";

export default function CreationPanel() {
  
  return (
    <div className={CreationPaneMain}>

      <BasicCreation></BasicCreation>

      <div className="advanced-create">
        <AdvancedSettings></AdvancedSettings>
        <ImageModifiers></ImageModifiers>
      </div>
    </div>
  );
}