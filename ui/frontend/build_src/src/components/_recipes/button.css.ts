// would prefer to use a var here, but it doesn't work
// vars: {
//   '--button-base-saturation': vars.colorMod.saturation.normal,
//   '--button-base-lightness': vars.colorMod.lightness.normal,
// },


import { recipe } from "@vanilla-extract/recipes";
import { vars } from "../../styles/theme/index.css";
// import { sprinkles } from "../../styles/sprinkles/index.css";


export const buttonStyle = recipe({

  base: {
    fontSize: vars.fonts.sizes.Subheadline,
    fontWeight: "bold",
    color: vars.colors.text.normal,
    padding: vars.spacing.small,
    border: "0",
    borderRadius: vars.trim.smallBorderRadius,
  },

  variants: {
    color: {
      primary: {
        // @ts-expect-error
        '--button-hue': vars.brandHue,
        '--button-base-saturation': vars.colorMod.saturation.normal,
        '--button-base-lightness': vars.colorMod.lightness.normal,
      },
      secondary: {
        // @ts-expect-error
        '--button-hue': vars.secondaryHue,
        '--button-base-saturation': vars.colorMod.saturation.normal,
        '--button-base-lightness': vars.colorMod.lightness.normal,
      },
      tertiary: {
        // @ts-expect-error
        '--button-hue': vars.tertiaryHue,
        '--button-base-saturation': vars.colorMod.saturation.normal,
        '--button-base-lightness': vars.colorMod.lightness.normal,
      },
      cancel: {
        // @ts-expect-error
        '--button-hue': vars.errorHue,
        '--button-base-saturation': vars.colorMod.saturation.normal,
        '--button-base-lightness': vars.colorMod.lightness.normal,
      },
      accent: {
        // @ts-expect-error
        '--button-hue': vars.backgroundAccentHue,
        '--button-base-saturation': vars.backgroundAccentSaturation,
        '--button-base-lightness': vars.backgroundAccentLightness,
      },
      clear: {
        backgroundColor: "transparent",
      },
    },

    type: {
      fill: {
        backgroundColor: `hsl(var(--button-hue),var(--button-base-saturation),${vars.colorMod.lightness.normal})`,
        border: `1px solid hsl(var(--button-hue),var(--button-base-saturation),${vars.colorMod.lightness.normal})`,
        ":hover": {
          backgroundColor: `hsl(var(--button-hue),${vars.colorMod.saturation.bright},${vars.colorMod.lightness.normal})`,
          border: `1px solid hsl(var(--button-hue),${vars.colorMod.saturation.bright},${vars.colorMod.lightness.normal})`,
        },
        ":active": {
          backgroundColor: `hsl(var(--button-hue),${vars.colorMod.saturation.bright},${vars.colorMod.lightness.dim})`,
          border: `1px solid hsl(var(--button-hue),${vars.colorMod.saturation.bright},${vars.colorMod.lightness.dim})`,
        },

        ":focus": {
          backgroundColor: `hsl(var(--button-hue),${vars.colorMod.saturation.bright},${vars.colorMod.lightness.dim})`,
          border: `1px solid hsl(var(--button-hue),${vars.colorMod.saturation.bright},${vars.colorMod.lightness.dim})`,
        },

        ":disabled": {
          backgroundColor: `hsl(var(--button-hue),${vars.colorMod.saturation.dim},${vars.colorMod.lightness.dim})`,
          border: `1px solid hsl(var(--button-hue),${vars.colorMod.saturation.dim},${vars.colorMod.lightness.dim})`,
        },
      },
      outline: {
        backgroundColor: "transparent",
        border: `1px solid hsl(var(--button-hue),var(--button-base-saturation),${vars.colorMod.lightness.normal})`,
        ":hover": {
          borderColor: `hsl(var(--button-hue),${vars.colorMod.saturation.bright},${vars.colorMod.lightness.normal})`,
        },

        ":active": {
          borderColor: `hsl(var(--button-hue),${vars.colorMod.saturation.bright},${vars.colorMod.lightness.dim})`,
        },

        ":focus": {
          borderColor: `hsl(var(--button-hue),${vars.colorMod.saturation.bright},${vars.colorMod.lightness.dim})`,
        },

        ":disabled": {
          borderColor: `hsl(var(--button-hue),${vars.colorMod.saturation.dim},${vars.colorMod.lightness.dim})`,
        },
      },
      action: {
        backgroundColor: "transparent",
        color: `hsl(var(--button-hue),var(--button-base-saturation),${vars.colorMod.lightness.normal})`,
        textDecoration: "underline",
        paddingLeft: 0,
        paddingRight: 0,
        ":hover": {
          color: `hsl(var(--button-hue),${vars.colorMod.saturation.bright},${vars.colorMod.lightness.normal})`,
        },

        ":active": {
          color: `hsl(var(--button-hue),${vars.colorMod.saturation.bright},${vars.colorMod.lightness.dim})`,
        },

        ":focus": {
          color: `hsl(var(--button-hue),${vars.colorMod.saturation.bright},${vars.colorMod.lightness.dim})`,
        },

        ":disabled": {
          color: `hsl(var(--button-hue),${vars.colorMod.saturation.dim},${vars.colorMod.lightness.dim})`,
        },
      }
    },

    size: {
      slim: {
        padding: vars.spacing.min,
        fontSize: vars.fonts.sizes.Caption,
      },
      large: {
        width: "100%",
        fontSize: vars.fonts.sizes.Headline,
      }
    }
  },

  defaultVariants: {
    color: "primary",
    type: "fill",
  },

});



// export const buttonRecipe = recipe({
//   base: {
//     fontSize: vars.fonts.sizes.Subheadline,
//     fontWeight: "bold",
//     color: vars.colors.text.normal,
//     padding: vars.spacing.small,
//     border: "0",
//     borderRadius: vars.trim.smallBorderRadius,
//   },

//   variants: {
//     color: {
//       primary: {
//         ...sprinkles({
//           backgroundColor: {
//             default: 'brandDefault',
//             hover: 'brandBright',
//             focus: 'brandFocus',
//             active: 'brandFocus',
//             disabled: 'brandDim',
//           },
//         })
//       }
//     }
//   }

// });