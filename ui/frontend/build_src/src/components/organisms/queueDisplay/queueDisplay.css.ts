import { style, globalStyle } from "@vanilla-extract/css";


import { vars } from "../../../styles/theme/index.css";

export const QueueDisplayMain = style({
  display: "flex",
  flexDirection: "column",
  width: '100%',
  height: '100%',
});

export const QueueListButtons = style({
  display: "flex",
  flexDirection: "row",
  justifyContent: "space-between",
  alignItems: "center",
  marginBottom: vars.spacing.large,
  marginTop: vars.spacing.medium,
});

globalStyle(`${QueueListButtons} button`, {
  flexGrow: 1,
});

globalStyle(`${QueueListButtons} button:first-child`, {
  marginRight: vars.spacing.medium,
});