import { style } from '@vanilla-extract/css';
import { recipe } from "@vanilla-extract/recipes";
import { vars } from "../../styles/theme/index.css";

export const tabStyles = recipe({
  base: {
    background: vars.backgroundMain,
    color: vars.colors.text.normal,
    padding: vars.spacing.small,
    borderRadius: `${vars.trim.smallBorderRadius} ${vars.trim.smallBorderRadius} 0 0`,
    border: `1px solid ${vars.backgroundLight}`,
    borderBottom: 'none',
    marginLeft: vars.spacing.small,
    boxShadow: `0px -1px 4px -2px ${vars.backgroundAccentMain} inset`,
    width: 'fit-content',
    ':focus': {
      outline: 'none',
    },
  },

  variants: {
    selected: {
      true: {
        background: vars.backgroundLight,
        color: vars.colors.text.normal,
        boxShadow: `0px -1px 4px -2px ${vars.backgroundDark} inset`,
      }
    }
  }
});

export const tabPanelStyles = recipe({
  base: {
    color: vars.colors.text.normal,
    // borderRadius: `0 0 ${vars.trim.smallBorderRadius} ${vars.trim.smallBorderRadius}`,
    background: vars.backgroundLight,
    padding: vars.spacing.medium,
    flexGrow: 1,
    overflow: 'auto',
    // "::-webkit-scrollbar": {
    //   width: "4px",
    // },

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
  },
});

