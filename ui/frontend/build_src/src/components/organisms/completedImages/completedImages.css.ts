import { style, globalStyle } from "@vanilla-extract/css";

import { vars } from "@styles/theme/index.css";
import {
  card as cardStyle,
} from '../../_recipes/card.css'

export const completedImagesMain = style({
  position: "relative",
});

// globalStyle(`${completedImagesMain} > button`, {
//   position: "absolute",
//   top: '-29px',
// });

export const completedImagesContent = style([cardStyle(), {
  height: "250px",
  width: "100%",
  display: "flex",
  padding: vars.spacing.medium,
  borderRadius: 0,
}]);

export const completedImagesList = style({
  display: "flex",
  flexDirection: "row",
  flexWrap: "nowrap",
  height: "100%",
  width: "100%",
  overflow: "auto",
  paddingLeft: vars.spacing.none,
});

globalStyle(`${completedImagesContent} li`, {
  position: "relative",
});

globalStyle(`${completedImagesContent} > li:first-of-type`, {
  marginLeft: vars.spacing.medium,
});

globalStyle(`${completedImagesContent} > li:last-of-type`, {
  marginRight: 0,
});

export const imageContain = style({
  width: "206px",
  backgroundColor: "black",
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  flexShrink: 0,
  border: "0 none",
  padding: "0",
  marginLeft: vars.spacing.medium,
  cursor: "pointer",
});

globalStyle(`${imageContain} img`, {
  width: "100%",
  objectFit: "contain",
});
