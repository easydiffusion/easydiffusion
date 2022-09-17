import React, { ChangeEvent } from "react";

import AdvancedSettings from "./advancedSettings";
import ImageModifiers from "./imageModifiers";
import InpaintingPanel from "./inpaintingPanel";

import { useImageCreate } from "../../../stores/imageCreateStore";

import "./creationPanel.css";

import {
  CreationPaneMain,
  InpaintingSlider
} from // @ts-ignore
  "./creationpane.css.ts";

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
