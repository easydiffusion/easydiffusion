import { recipe } from "@vanilla-extract/recipes";
import { vars } from "../../styles/theme/index.css";

export const buttonStyle = recipe({

  // would prefer to use a var here, but it doesn't work
  // vars: {
  //   '--button-base-saturation': vars.colorMod.saturation.normal,
  //   '--button-base-lightness': vars.colorMod.lightness.normal,
  // },

  base: {
    fontSize: vars.fonts.sizes.Subheadline,
    fontWeight: "bold",
    color: vars.colors.text.normal,
    padding: vars.spacing.small,
    border: "0",
    borderRadius: vars.trim.smallBorderRadius,
  },

  variants: {
    type: {
      primary: {
        '--primary-button-hue': vars.brandHue,
        backgroundColor: `hsl(var(--primary-button-hue),${vars.colorMod.saturation.normal},${vars.colorMod.lightness.normal})`,
        ":hover": {
          backgroundColor: `hsl(var(--primary-button-hue),${vars.colorMod.saturation.bright},${vars.colorMod.lightness.normal})`,
        },

        ":active": {
          backgroundColor: `hsl(var(--primary-button-hue),${vars.colorMod.saturation.bright},${vars.colorMod.lightness.dim})`,
        },

        ":focus": {
          backgroundColor: `hsl(var(--primary-button-hue),${vars.colorMod.saturation.bright},${vars.colorMod.lightness.dim})`,
        },

        ":disabled": {
          backgroundColor: `hsl(var(--primary-button-hue),${vars.colorMod.saturation.dim},${vars.colorMod.lightness.dim})`,
        },
      },
      secondary: {
        '--secondary-button-hue': vars.secondaryHue,
        backgroundColor: `hsl(var(--secondary-button-hue),${vars.colorMod.saturation.normal},${vars.colorMod.lightness.normal})`,
        ":hover": {
          backgroundColor: `hsl(var(--secondary-button-hue),${vars.colorMod.saturation.bright},${vars.colorMod.lightness.normal})`,
        },

        ":active": {
          backgroundColor: `hsl(var(--secondary-button-hue),${vars.colorMod.saturation.bright},${vars.colorMod.lightness.dim})`,
        },

        ":focus": {
          backgroundColor: `hsl(var(--secondary-button-hue),${vars.colorMod.saturation.bright},${vars.colorMod.lightness.dim})`,
        },

        ":disabled": {
          backgroundColor: `hsl(var(--secondary-button-hue),${vars.colorMod.saturation.dim},${vars.colorMod.lightness.dim})`,
        },
      },
      tertiary: {
        '--tertiary-button-hue': vars.tertiaryHue,
        backgroundColor: `hsl(var(--tertiary-button-hue),${vars.colorMod.saturation.normal},${vars.colorMod.lightness.normal})`,
        ":hover": {
          backgroundColor: `hsl(var(--tertiary-button-hue),${vars.colorMod.saturation.bright},${vars.colorMod.lightness.normal})`,
        },

        ":active": {
          backgroundColor: `hsl(var(--tertiary-button-hue),${vars.colorMod.saturation.bright},${vars.colorMod.lightness.dim})`,
        },

        ":focus": {
          backgroundColor: `hsl(var(--tertiary-button-hue),${vars.colorMod.saturation.bright},${vars.colorMod.lightness.dim})`,
        },

        ":disabled": {
          backgroundColor: `hsl(var(--tertiary-button-hue),${vars.colorMod.saturation.dim},${vars.colorMod.lightness.dim})`,
        },
      },
      cancel: {
        '--cancel-button-hue': vars.errorHue,
        backgroundColor: `hsl(var(--cancel-button-hue), ${vars.colorMod.saturation.normal},${vars.colorMod.lightness.normal})`,
        ":hover": {
          backgroundColor: `hsl(var(--cancel-button-hue), ${vars.colorMod.saturation.bright},${vars.colorMod.lightness.normal})`,
        },
        ":active": {
          backgroundColor: `hsl(var(--cancel-button-hue), ${vars.colorMod.saturation.bright} ,${vars.colorMod.lightness.dim})`,
        },
        ":focus": {
          backgroundColor: `hsl(var(--cancel-button-hue), ${vars.colorMod.saturation.bright} ,${vars.colorMod.lightness.dim})`,
        },
        ":disabled": {
          backgroundColor: `hsl(var(--cancel-button-hue), ${vars.colorMod.saturation.dim} ,${vars.colorMod.lightness.dim})`,
        },
      },
      clear: {
        backgroundColor: "transparent",
      },
    },

    size: {
      large: {
        width: "100%",
        fontSize: vars.fonts.sizes.Headline,
      }
    }
  },
  defaultVariants: {
    type: "primary",
  },

});