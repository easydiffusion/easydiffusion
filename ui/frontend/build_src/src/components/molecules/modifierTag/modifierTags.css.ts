import { style, globalStyle } from "@vanilla-extract/css";
import { vars } from '../../../styles/theme/index.css';

import { card } from '../../_recipes/card.css';

export const ModifierTagMain = style([
  card({
    backing: 'normal',
    level: 1,
    info: true
  }), {
    position: "relative",
    width: "fit-content",
    borderColor: `hsl(${vars.brandHue}, ${vars.colorMod.saturation.normal}, ${vars.colorMod.lightness.normal})`,
    padding: vars.spacing.small,
  }
]);

globalStyle(`${ModifierTagMain}.selected`, {
  backgroundColor: "rgb(131, 11, 121)",
})

globalStyle(`${ModifierTagMain} p`, {
  margin: 0,
  textAlign: "center",
  marginBottom: "2px",
});


export const TagText = style({
  opacity: 1,
});

export const TagToggle = style({
  opacity: 0.3,
});


export const ModifierActions = style({
  position: "absolute",
  top: "0",
  left: "0",
  height: "100%",
  width: "100%",
  display: "flex",
  flexDirection: "row",
});

globalStyle(`${ModifierActions} button`, {
  flexGrow: 1,
  backgroundColor: "transparent",
  border: "none",
  boxShadow: `inset 0 0 24px 0px rgb(255 255 255 / 50%)`,
  borderRadius: "5px",
  padding: "0",
});

export const tagPreview = style({
  display: 'flex',
  justifyContent: 'center',
});

globalStyle(`${tagPreview} img`, {
  width: "90px",
  height: "100%",
  objectFit: "cover",
  objectPosition: "center",
});

