import { style, globalStyle } from "@vanilla-extract/css";


import { vars } from "../../../../styles/theme/index.css";


import { QueueStatus } from "../../../../stores/requestQueueStore";


import {
  card
} from '../../../_recipes/card.css';

export const QueueItemMain = style([card(
  {
    baking: "dark",
    level: 1
  }
), {
  display: "flex",
  flexDirection: "column",
  width: "100%",
  marginBottom: vars.spacing.medium,
}]);

export const QueueItemInfo = style({

});

globalStyle(`${QueueItemInfo} p`, {
  marginBottom: vars.spacing.small,
});

globalStyle(`${QueueItemMain}.${QueueStatus.processing}`, {
  backgroundColor: `hsl(${vars.secondaryHue}, ${vars.colorMod.saturation.normal}, ${vars.colorMod.lightness.normal})`,
});

globalStyle(`${QueueItemMain}.${QueueStatus.pending}`, {
  backgroundColor: vars.backgroundDark,
  // `hsl(${vars.warningHue}, ${vars.colorMod.saturation.normal}, ${vars.colorMod.lightness.normal})`,
});

globalStyle(`${QueueItemMain}.${QueueStatus.paused}`, {
  backgroundColor: vars.backgroundDark,
  //`hsl(${vars.tertiaryHue}, ${vars.colorMod.saturation.normal}, ${vars.colorMod.lightness.normal})`,
});

globalStyle(`${QueueItemMain}.${QueueStatus.complete}`, {
  backgroundColor: `hsl(${vars.successHue}, ${vars.colorMod.saturation.normal}, ${vars.colorMod.lightness.normal})`,
});

globalStyle(`${QueueItemMain}.${QueueStatus.error}`, {
  backgroundColor: `hsl(${vars.errorHue}, ${vars.colorMod.saturation.normal}, ${vars.colorMod.lightness.normal})`,
});

export const QueueButtons = style({
  display: "flex",
  flexDirection: "row",
  justifyContent: "space-between",
  alignItems: "center",
});