import React, { ChangeEvent } from "react";

import AdvancedSettings from "./advancedSettings";
import ImageModifiers from "./imageModifiers";
import InpaintingPanel from "./inpaintingPanel";

// this works but causes type errors so its not worth it for now
// import { useImageCreate } from "@stores/imageCreateStore.ts";

import { useImageCreate } from "../../../stores/imageCreateStore";

import "./creationPanel.css";

import {
  CreationPaneMain,
  InpaintingSlider, // @ts-expect-error
} from "./creationpane.css.ts";

import BasicCreation from "./basicCreation";

export default function CreationPanel() {
  const isInPaintingMode = useImageCreate((state) => state.isInpainting);
  return (
    <>
      <div className={CreationPaneMain}>
        <BasicCreation></BasicCreation>
        <AdvancedSettings></AdvancedSettings>
        <ImageModifiers></ImageModifiers>
      </div>

      {isInPaintingMode && (
        <div className={InpaintingSlider}>
          <InpaintingPanel></InpaintingPanel>
        </div>
      )}
    </>
  );
}
