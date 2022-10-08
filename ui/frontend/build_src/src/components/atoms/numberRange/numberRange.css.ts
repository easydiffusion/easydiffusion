import { style } from '@vanilla-extract/css';
import { vars } from '../../../styles/theme/index.css'
export const NumberRangeRoot = style({
  display: 'flex',
  marginRight: vars.spacing.small,
});


export const NumberRangeInput = style({
  margin: 0,
  padding: 0,
  border: 0,
  marginRight: vars.spacing.small,
  //WebkitAppearance: 'none',

  // ':hover': {
  //   backgroundColor: 'red'
  // },

  '::-webkit-slider-runnable-track': {
    // WebkitAppearance: 'none',
    backgroundColor: vars.backgroundDark,
    borderRadius: vars.trim.smallBorderRadius,
    // height: '10px',
  },

  '::-webkit-slider-thumb': {
    WebkitAppearance: 'none',
    color: 'red',
    //`hsl(${vars.brandHue}, ${vars.colorMod.saturation.normal},${vars.colorMod.lightness.normal})`,
  },

});