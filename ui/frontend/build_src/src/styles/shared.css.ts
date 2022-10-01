import { style, globalStyle } from "@vanilla-extract/css";
import { vars } from "./theme/index.css";


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

