import { style, globalStyle } from "@vanilla-extract/css";
import { vars } from "@styles/theme/index.css";

export const AppLayout = style({
  position: "relative",

  width: "100vw",
  height: "100vh",
  pointerEvents: "auto",
  display: "grid",
  backgroundColor: vars.backgroundMain,
  gridTemplateColumns: "400px 1fr",
  gridTemplateRows: "45px 1fr 115px",
  gridTemplateAreas: `
    "create header header"
    "create display display"
    "create display display"
  `,
});

export const HeaderLayout = style({
  gridArea: "header",
});

export const CreateLayout = style({
  gridArea: "create",
  position: "relative",
  display: "flex",
  flexDirection: "column",
});

export const DisplayLayout = style({
  gridArea: "display",
  overflow: "auto",
});

export const FooterLayout = style({
  gridArea: "footer",
  display: "flex",
  justifyContent: "center",
});
