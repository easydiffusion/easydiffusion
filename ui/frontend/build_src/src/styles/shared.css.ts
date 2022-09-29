import { style, globalStyle } from "@vanilla-extract/css";
import { vars } from "./theme/index.css";

export const PanelBox = style({
  background: vars.colors.backgroundAlt,
  color: vars.colors.text.normal,
  padding: vars.spacing.medium,
  borderRadius: vars.trim.smallBorderRadius,
  marginBottom: vars.spacing.medium,
  // TODO move this to the theme
  boxShadow:
    "0 4px 8px 0 rgba(0, 0, 0, 0.15), 0 6px 20px 0 rgba(0, 0, 0, 0.15)",
});

globalStyle(`${PanelBox} .panel-box-toggle-btn`, {
  display: "block",
  width: "100%",
  textAlign: "left",
  backgroundColor: "transparent",
  color: vars.colors.text.normal,
  border: "0 none",
  cursor: "pointer",
  padding: "0",
});

//TODO this should probably just be for all li elements
export const SettingItem = style({
  marginBottom: vars.spacing.medium,

  selectors: {
    "&:last-of-type": {
      marginBottom: vars.spacing.none,
    },
  },

});


export const IconFont = style({
  // reliant on font-awesome cdn
  fontFamily: "Font Awesome 6 Free"
});


export const MenuButton = style({
  display: "block",
  width: "100%",
  textAlign: "left",
  backgroundColor: "transparent",
  color: vars.colors.text.normal,
  border: "0 none",
  cursor: "pointer",
  padding: "0",
  marginBottom: vars.spacing.medium,
});

globalStyle(`${MenuButton}> h4`, {
  color: "#e7ba71",
});


export const BrandedButton = style({
  backgroundColor: vars.colors.brand,
  fontSize: vars.fonts.sizes.Subheadline,
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
});
