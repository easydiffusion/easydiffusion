import {
  defineProperties,
  createSprinkles
} from '@vanilla-extract/sprinkles';

import { vars } from '../theme/index.css';

const hues = {
  brand: vars.brandHue,
  secondary: vars.secondaryHue,
  tertiary: vars.tertiaryHue,
  error: vars.errorHue,
  warning: vars.warningHue,
  success: vars.successHue,
};

const saturation = {
  bright: vars.colorMod.saturation.bright,
  normal: vars.colorMod.saturation.normal,
  dimmed: vars.colorMod.saturation.dimmed,
  dim: vars.colorMod.saturation.dim,
};

const lightness = {
  normal: vars.colorMod.lightness.normal,
  bright: vars.colorMod.lightness.bright,
  dim: vars.colorMod.lightness.dim,
};

const colors = {
  brandDefault: `hsl(${hues.brand},${saturation.normal},${lightness.normal})`,
  brandBright: `hsl(${hues.brand},${saturation.bright},${lightness.normal})`,
  brandFocus: `hsl(${hues.brand},${saturation.bright},${lightness.dim})`,
  brandDim: `hsl(${hues.brand},${saturation.dim},${lightness.dim})`,

  secondaryDefault: `hsl(${hues.secondary},${saturation.normal},${lightness.normal})`,
  secondaryBright: `hsl(${hues.secondary},${saturation.bright},${lightness.normal})`,
  secondaryFocus: `hsl(${hues.secondary},${saturation.bright},${lightness.dim})`,
  secondaryDim: `hsl(${hues.secondary},${saturation.dim},${lightness.dim})`,

  tertiaryDefault: `hsl(${hues.tertiary},${saturation.normal},${lightness.normal})`,
  tertiaryBright: `hsl(${hues.tertiary},${saturation.bright},${lightness.normal})`,
  tertiaryFocus: `hsl(${hues.tertiary},${saturation.bright},${lightness.dim})`,
  tertiaryDim: `hsl(${hues.tertiary},${saturation.dim},${lightness.dim})`,
};

const stateProperties = defineProperties({
  properties: {
    backgroundColor: colors,
    borderColor: colors,
    color: colors,
  },
  conditions: {
    default: {},
    hover: { selector: '&:hover' },
    focus: { selector: '&:focus' },
    active: { selector: '&:active' },
    disabled: { selector: '&:disabled' },
  },
  defaultCondition: 'default',
  shorthands: {
    background: ['backgroundColor', 'borderColor'],
    text: ['color'],
  },
});


export const sprinkles = createSprinkles(stateProperties);
