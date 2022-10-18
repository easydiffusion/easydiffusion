import { globalStyle } from "@vanilla-extract/css";
import { recipe } from "@vanilla-extract/recipes";
import { vars } from "@styles/theme/index.css";



export const DisclosureMain = recipe({
  base: {
    position: 'relative',
    display: 'flex',
    flexDirection: 'column',
    width: '100%',
  },
  variants: {
    open: {
      true: {
        zIndex: '1',
      },
    },
  },
  defaultVariants: {
    open: false,
  },
});

export const DisclosureButtonStyle = recipe({
  base: {
    backgroundColor: "transparent",
    border: "0 none",
    cursor: "pointer",
    padding: vars.spacing.none,
    fontSize: vars.fonts.sizes.Subheadline,
  },
  variants: {
    open: {
      true: {
        color: vars.colors.primary,
      },
    },
  },
  defaultVariants: {
    open: false,
  },
});

globalStyle(`${DisclosureButtonStyle} > i`, {
  marginRight: vars.spacing.small,
});

export const DisclosurePanelMain = recipe({
  base: {
    position: 'absolute',
    top: '100%',
    right: '0',
    zIndex: '1',
  },
  variants: {
    open: {
      true: {
        zIndex: '1',
      },
    },
  },
  defaultVariants: {
    open: false,
  },
});




