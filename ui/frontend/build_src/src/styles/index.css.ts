import { globalStyle } from "@vanilla-extract/css";
import { vars } from "./theme/index.css";

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

globalStyle(`.visually-hidden`, {
  visibility: "hidden",
  position: "absolute",
});

/** RESETS */
globalStyle(`h1, h2, h3, h4, h5, h6, p, label, ul, textarea`, {
  margin: 0,
  padding: 0,
  fontFamily: vars.fonts.body,
});

globalStyle(`h3`, {
  fontSize: vars.fonts.sizes.Subheadline,
});

globalStyle(`h4, h5`, {
  fontSize: vars.fonts.sizes.SubSubheadline,
});

globalStyle(`h6`, {
  fontSize: vars.fonts.sizes.Body,
});

globalStyle(`p, label`, {
  fontSize: vars.fonts.sizes.Body,
});

globalStyle(`textarea`, {
  padding: 0,
  border: "none",
  fontSize: vars.fonts.sizes.Body,
  fontWeight: "bold",
});

globalStyle(`a`, {
  color: vars.colors.link,
  textDecoration: "none",
});