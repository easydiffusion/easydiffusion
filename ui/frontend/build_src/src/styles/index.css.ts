import { globalStyle } from "@vanilla-extract/css";
// @ts-expect-error
import { vars } from "./theme/index.css.ts";

// baisc body style
globalStyle("body", {
  margin: 0,
  minWidth: "320px",
  minHeight: "100vh",
});

// single page style
globalStyle("#root", {
  position: "absolute",
  top: 0,
  left: 0,
  width: "100vw",
  height: "100vh",
  overflow: "hidden",
});

// border box all
globalStyle(`*`, {
  boxSizing: "border-box",
});

globalStyle(`button`, {
  fontSize: vars.fonts.sizes.Body,
});

/** RESETS */
globalStyle(`p, h1, h2, h3, h4, h5, h6, ul`, {
  margin: 0,
});

globalStyle(`h3`, {
  fontSize: vars.fonts.sizes.Subheadline,
  fontFamily: vars.fonts.body,
});

globalStyle(`h4, h5`, {
  fontSize: vars.fonts.sizes.SubSubheadline,
  fontFamily: vars.fonts.body,
});

globalStyle(`p, label`, {
  fontSize: vars.fonts.sizes.Body,
  fontFamily: vars.fonts.body,
});

globalStyle(`textarea`, {
  margin: 0,
  padding: 0,
  border: "none",
  fontSize: vars.fonts.sizes.Body,
  fontWeight: "bold",
  fontFamily: vars.fonts.body,
});
