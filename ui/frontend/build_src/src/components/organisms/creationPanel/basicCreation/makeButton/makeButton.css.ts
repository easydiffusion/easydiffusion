import { style } from "@vanilla-extract/css";

export const MakeButtonStyle = style({
  width: "100%",
  backgroundColor: "rgb(38, 77, 141)",
  fontSize: "1.5em",
  fontWeight: "bold",
  color: "white",
  padding: "8px",
  borderRadius: "5px",

  ":disabled": {
    backgroundColor: "rgb(38, 77, 141, 0.5)",
  },
});
