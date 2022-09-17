import { style } from "@vanilla-extract/css";
// @ts-ignore
import { vars } from "../../styles/theme.css.ts";

export const AppLayout = style({
  position: "relative",
  backgroundColor: vars.colors.background,
  width: "100%",
  height: "100%",
  pointerEvents: "auto",
  display: "grid",
  // backgroundColor: "rgb(32, 33, 36)",
  gridTemplateColumns: "400px 1fr",
  gridTemplateRows: "100px 1fr 50px",
  gridTemplateAreas: `
    "header header header"
    "create display display"
    "create footer footer"
  `,

  "@media": {
    "screen and (max-width: 800px)": {
      gridTemplateColumns: "1fr",
      gridTemplateRows: "100px 1fr 1fr 50px",
      gridTemplateAreas: `
        "header"
        "create"
        "display"
        "footer"
      `,
    },
  },
});

export const HeaderLayout = style({
  gridArea: "header",
});

export const CreateLayout = style({
  gridArea: "create",
  position: "relative",
});

export const DisplayLayout = style({
  gridArea: "display",
  overflow: "auto",
});

export const FooterLayout = style({
  gridArea: "footer",
});
