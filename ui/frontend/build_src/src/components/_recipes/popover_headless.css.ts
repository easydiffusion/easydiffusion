
import { style, globalStyle } from "@vanilla-extract/css";

// import { recipe } from "@vanilla-extract/recipes";
import { vars } from "@styles/theme/index.css";

import {
  card
} from "./card.css";

export const PopoverMain = style({
  position: 'relative'
});

export const PopoverButtonStyle = style({
  backgroundColor: "transparent",
  border: "0 none",
  cursor: "pointer",
  padding: vars.spacing.none,
  fontSize: vars.fonts.sizes.Subheadline,
});

globalStyle(`${PopoverButtonStyle} > i`, {
  marginRight: vars.spacing.small,
});

export const PopoverPanelMain = style([card(
  {
    backing: 'dark',
    level: 2,
  }
), {
  position: 'absolute',
  top: '100%',
  right: '0',
  zIndex: '1',
}]);


