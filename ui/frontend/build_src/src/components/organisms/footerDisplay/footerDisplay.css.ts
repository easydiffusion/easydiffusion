import { style, globalStyle } from "@vanilla-extract/css";

// @ts-expect-error
import { vars } from "../../../styles/theme/index.css.ts";

export const FooterDisplayMain = style({
  color: vars.colors.text.normal,
  fontSize: vars.fonts.sizes.Caption,

  display: "inline-block",
  // marginTop: vars.spacing.medium,
  // marginBottom: vars.spacing.medium,
  // TODO move this to the theme
  padding: vars.spacing.small,
  boxShadow:
    "0 4px 8px 0 rgba(0, 0, 0, 0.15), 0 6px 20px 0 rgba(0, 0, 0, 0.15)",
});

export const CoffeeButton = style({
  height: "23px",
  transform: "translateY(25%)",
});

globalStyle(`${FooterDisplayMain} a`, {
  color: vars.colors.link,
  textDecoration: "none",
});

globalStyle(`${FooterDisplayMain} a:hover`, {
  textDecoration: "underline",
});

globalStyle(`${FooterDisplayMain} a:visited`, {
  color: vars.colors.link,
});

globalStyle(`${FooterDisplayMain} a:active`, {
  color: vars.colors.link,
});

globalStyle(`${FooterDisplayMain} a:focus`, {
  color: vars.colors.link,
});

globalStyle(`${FooterDisplayMain} p`, {
  margin: vars.spacing.min,
});

// .footer-display {
//   color: #ffffff;
//   display: flex;
//   flex-direction: column;
//   align-items: center;
//   justify-content: center;
// }

// #coffeeButton {
//   height: 23px;
//   transform: translateY(25%);
// }
