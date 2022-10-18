import { style, globalStyle } from '@vanilla-extract/css';
import { vars } from "../../../styles/theme/index.css";

export const CheckMain = style({
  display: 'flex',
  alignItems: 'center',
});

globalStyle(`${CheckMain} >:first-child`, {
  marginRight: '10px'
});

export const CheckContent = style({

  background: vars.backgroundDark,

  width: '1.5em',
  height: '1.5em',
  border: 0,
  position: 'relative',
  display: 'inline-flex',
  padding: 0,
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',

  borderRadius: vars.trim.inputBorderRadius,

  // ':hover': {

  //0 0 6px -4px white inset
  selectors: {
    [`${CheckMain}[data-disabled="true"] &`]: {
      background: vars.backgroundLight,
    },
    [`&[data-headlessui-state="checked"]`]: {
      boxShadow: `0 0 5px -1px var(--backgroundAccentMain__4vfmtjw) inset`,
    }
  }
});


export const CheckInner = style({
  color: vars.backgroundAccentMain,

  selectors: {
    [`${CheckContent}[data-headlessui-state="checked"] &`]: {
      color: `hsl(${vars.brandHue}, ${vars.backgroundAccentSaturation}, ${vars.backgroundAccentLightness})`,
    },
    [`${CheckMain}[data-disabled="true"] ${CheckContent}[data-headlessui-state="checked"] &`]: {
      color: vars.backgroundAccentMain,
    },
  }
});
