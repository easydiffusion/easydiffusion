import { style, globalStyle } from '@vanilla-extract/css';
import { vars } from "../../../styles/theme/index.css";

export const CheckMain = style({
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
  selectors: {
    [`${CheckMain}[data-disabled="true"] &`]: {
      background: vars.backgroundLight,
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
