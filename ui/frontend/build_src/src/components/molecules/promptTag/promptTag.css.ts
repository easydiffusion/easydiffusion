/* eslint-disable @typescript-eslint/restrict-template-expressions */
import { style, globalStyle } from '@vanilla-extract/css';

import { XButton } from "@styles/shared.css";
import { vars } from '@styles/theme/index.css';
import { card } from '../../_recipes/card.css';


export const PromptTagMain = style([
  card({
    backing: 'normal',
    level: 1,
    info: true
  }), {
    position: "relative",
    width: "fit-content",
    backgroundColor: `hsl(${vars.backgroundLight}, ${vars.colorMod.saturation.normal}, ${vars.colorMod.lightness.normal})`,
    padding: vars.spacing.small,
  }
]);

export const PromptTagText = style({
  opacity: 1,
  fontSize: vars.fonts.sizes.Plain,
});

export const PromptTagToggle = style({
  opacity: 0.3,
  fontSize: vars.fonts.sizes.Plain,
});

globalStyle(`${PromptTagMain}.positive`, {
  borderColor: `hsl(${vars.brandHue}, ${vars.colorMod.saturation.normal}, ${vars.colorMod.lightness.normal})`,
});

globalStyle(`${PromptTagMain}.negative`, {
  borderColor: `hsl(${vars.errorHue}, ${vars.colorMod.saturation.normal}, ${vars.colorMod.lightness.normal})`,
});

export const TagToggle = style({
  position: "absolute",
  top: "0",
  right: "0",
  height: "100%",
  width: "100%",
  border: "none",
  backgroundColor: "transparent",
  boxShadow: `inset 0 0 24px 0px rgb(255 255 255 / 50%)`,
});

export const TagRemoveButton = style([XButton, {
  top: '-4px',
  left: '4px',
  padding: '0',
}]);