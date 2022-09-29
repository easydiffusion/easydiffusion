import { style } from "@vanilla-extract/css";
import { PanelBox } from "../../../styles/shared.css";
export const CreationPaneMain = style({
  position: "relative",
  width: "100%",
  height: "100%",
  padding: "0 10px",
  overflowY: "auto",
  overflowX: "hidden",
});

export const InpaintingSlider = style({
  position: "absolute",
  top: "10px",
  left: "400px",
  zIndex: 1,
  backgroundColor: "rgba(0, 0, 0, 0.5)",
});

export const QueueSlider = style([PanelBox, {
  position: "absolute",
  top: "10px",
  left: "400px",
  zIndex: 1,
  maxHeight: "90%",
  overflowY: "auto",
}]);