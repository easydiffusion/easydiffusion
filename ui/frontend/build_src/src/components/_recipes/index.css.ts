import { recipe } from "@vanilla-extract/recipes";
import { vars } from "../../styles/theme/index.css";


export const card = recipe({
  base: {
    background: vars.colors.background,
    color: vars.colors.text.normal,
    padding: vars.spacing.medium,
    borderRadius: vars.trim.smallBorderRadius,
  },
  variants: {
    level: {
      1: { boxShadow: "0 4px 8px 0 rgba(0, 0, 0, 0.15), 0 6px 20px 0 rgba(0, 0, 0, 0.15)" },
      2: { boxShadow: "0 4px 8px 0 rgba(0, 0, 0, 0.15), 0 6px 20px 0 rgba(0, 0, 0, 0.15)" },
      3: { boxShadow: "0 4px 8px 0 rgba(0, 0, 0, 0.15), 0 6px 20px 0 rgba(0, 0, 0, 0.15)" },
    },
  },
  defaultVariants: {
    level: 1,
  },
});


