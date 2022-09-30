import { style, globalStyle } from "@vanilla-extract/css";


import { vars } from "../../../../styles/theme/index.css";

import { BrandedButton } from "../../../../styles/shared.css";

import { QueueStatus } from "../../../../stores/requestQueueStore";

export const QueueItemMain = style({
  display: "flex",
  flexDirection: "column",
  width: "100%",
  padding: vars.spacing.small,
  borderRadius: vars.trim.smallBorderRadius,
  marginBottom: vars.spacing.medium,
  boxShadow: "0 4px 8px 0 rgba(0, 0, 0, 0.15), 0 6px 20px 0 rgba(0, 0, 0, 0.15)",
});

export const QueueItemInfo = style({

});

globalStyle(`${QueueItemInfo} p`, {
  marginBottom: vars.spacing.small,
});


globalStyle(`${QueueItemMain}.${QueueStatus.processing}`, {
  backgroundColor: vars.colors.warning,
});

globalStyle(`${QueueItemMain}.${QueueStatus.pending}`, {
  backgroundColor: vars.colors.backgroundDark,
});

globalStyle(`${QueueItemMain}.${QueueStatus.paused}`, {
  backgroundColor: vars.colors.backgroundAlt,
});

globalStyle(`${QueueItemMain}.${QueueStatus.complete}`, {
  backgroundColor: vars.colors.success,
});

globalStyle(`${QueueItemMain}.${QueueStatus.error}`, {
  backgroundColor: vars.colors.error,
});

export const QueueButtons = style({
  display: "flex",
  flexDirection: "row",
  justifyContent: "space-between",
  alignItems: "center",
});


// TODO these should be a button recipe?
// export const CompleteButtton = style([BrandedButton, {

// }]);

// export const PauseButton = style([BrandedButton, {

// }]);

// export const ResumeButton = style([BrandedButton, {

// }]);

// export const CancelButton = style([BrandedButton, {

// }]);

// export const RetryButton = style([BrandedButton, {

// }]);

// export const SendToTopButton = style([BrandedButton, {

// }]);
