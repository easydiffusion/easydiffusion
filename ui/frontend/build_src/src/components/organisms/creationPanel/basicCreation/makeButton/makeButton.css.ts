import { style } from "@vanilla-extract/css";

// @ts-expect-error
import { vars } from "../../../../../styles/theme/index.css.ts";

export const MakeButtonStyle = style({
  width: "100%",
  backgroundColor: vars.colors.brand,
  fontSize: vars.fonts.sizes.Headline,
  fontWeight: "bold",
  color: vars.colors.text.normal,
  padding: vars.spacing.small,
  borderRadius: vars.trim.smallBorderRadius,

  ":hover": {
    backgroundColor: vars.colors.brandHover,
  },

  ":active": {
    backgroundColor: vars.colors.brandActive,
  },

  ":disabled": {
    backgroundColor: vars.colors.brandDimmed,
    color: vars.colors.text.dimmed,
  },

  ":focus": {
    outline: "none",
  },
});
