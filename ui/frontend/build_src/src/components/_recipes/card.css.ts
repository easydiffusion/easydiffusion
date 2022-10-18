import { recipe } from "@vanilla-extract/recipes";
import { vars } from "@styles/theme/index.css";

export const card = recipe({

  base: {
    color: vars.colors.text.normal,
    padding: vars.spacing.medium,
  },

  variants: {
    backing: {
      normal: {
        background: vars.backgroundMain,
      },
      light: {
        background: vars.backgroundLight,
      },
      dark: {
        background: vars.backgroundDark,
      },
    },

    rounded: {
      true: {
        borderRadius: vars.trim.smallBorderRadius,
      },
    },

    info: {
      true: {
        background: vars.backgroundDark,
        border: `1px solid ${vars.backgroundAccentMain}`,
      },
    },

    level: {
      down: { boxShadow: "0 4px 8px 0 rgba(0, 0, 0, 0.15), 0 6px 20px 0 rgba(0, 0, 0, 0.15) inset" },
      flat: {},
      1: { boxShadow: "0 4px 8px 0 rgba(0, 0, 0, 0.15), 0 6px 20px 0 rgba(0, 0, 0, 0.15)" },
      2: { boxShadow: "0 4px 8px 0 rgba(0, 0, 0, 0.15), 0 6px 20px 0 rgba(0, 0, 0, 0.15)" },
      3: { boxShadow: "0 4px 8px 0 rgba(0, 0, 0, 0.15), 0 6px 20px 0 rgba(0, 0, 0, 0.15)" },
    },
  },
  defaultVariants: {
    backing: "light",
    level: 'flat',
    rounded: true,
  },
});


