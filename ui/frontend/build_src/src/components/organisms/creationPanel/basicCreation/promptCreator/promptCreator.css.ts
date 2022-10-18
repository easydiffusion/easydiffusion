import { style, globalStyle } from '@vanilla-extract/css';

import { vars } from "../../../../../styles/theme/index.css";

import { sprinkles } from "../../../../../styles/sprinkles/index.css";
import {
  buttonStyle,
} from "../../../../_recipes/button.css";

export const PromptCreatorMain = style({
  display: 'flex',
  flexDirection: 'column',
  width: '100%',
  height: '100%',
  marginBottom: 0,
});

// alternative way to s
export const prmptBtn = style([buttonStyle({ size: 'slim' }), sprinkles({
  backgroundColor: {
    default: 'brandDefault',
    hover: 'brandBright',
    focus: 'brandFocus',
    active: 'brandFocus',
    disabled: 'brandDim',
  },
}), {}]);

globalStyle(`${PromptCreatorMain} textarea`, {
  width: '100%',
});

globalStyle(`${PromptCreatorMain} input`, {
  width: '100%',
});


globalStyle(`${PromptCreatorMain} > div`, {
  marginBottom: vars.spacing.small,
});

export const ToggleGroupMain = style({
  //  '--toggle-size': '30px',
});

export const ToggleMain = style({
  background: vars.backgroundDark,
  height: '22px',
  borderRadius: '15px',
  width: '34px',
  border: 0,
  position: 'relative',
  display: 'inline-flex',
  padding: 0,
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  boxShadow: `0 0 2px 0  ${vars.backgroundDark}`,
});

export const ToggleLabel = style({
});

export const ToggleEnabled = style({
});

globalStyle(`${ToggleMain}[data-headlessui-state="checked"]`, {
  background: vars.backgroundLight,
});

export const TogglePill = style({
  display: 'inline-flex',
  height: '18px',
  width: '30px',
  borderRadius: '15px',
  background: vars.backgroundDark,
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
});

globalStyle(`${ToggleMain}[data-headlessui-state="checked"] ${TogglePill}`, {
  background: vars.backgroundAccentMain,
});

globalStyle(`${TogglePill} p`, {
  color: vars.colors.text.normal,
});


export const inputRow = style({
  marginTop: vars.spacing.small,
  display: 'flex',
  flexDirection: 'row',
});

globalStyle(`${inputRow} > button`, {
  flexGrow: 1,
  marginRight: vars.spacing.medium,
});