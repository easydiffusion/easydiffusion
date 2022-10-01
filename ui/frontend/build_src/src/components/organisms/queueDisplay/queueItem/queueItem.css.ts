import { style, globalStyle } from "@vanilla-extract/css";


import { vars } from "../../../../styles/theme/index.css";


import { QueueStatus } from "../../../../stores/requestQueueStore";


import {
  card
} from '../../../_recipes/card.css';

export const QueueItemMain = style([card(
  {

    info: true,
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

globalStyle(`${QueueItemMain}.${QueueStatus.complete}`, {
  borderColor: `hsl(${vars.secondaryHue}, ${vars.colorMod.saturation.normal}, ${vars.colorMod.lightness.normal})`,
});

globalStyle(`${QueueItemMain}.${QueueStatus.processing}`, {
  borderColor: `hsl(${vars.tertiaryHue}, ${vars.colorMod.saturation.bright}, ${vars.colorMod.lightness.bright})`,
});

globalStyle(`${QueueItemMain}.${QueueStatus.pending}`, {
  borderColor: `hsl(${vars.backgroundAccentMain}, ${vars.colorMod.saturation.bright}, ${vars.colorMod.lightness.normal})`,
});

globalStyle(`${QueueItemMain}.${QueueStatus.paused}`, {
  borderColor: `hsl(${vars.backgroundAccentMain}, ${vars.colorMod.saturation.dim}, ${vars.colorMod.lightness.dim})`,
  backgroundColor: `hsl(${vars.backgroundAccentMain}, ${vars.colorMod.saturation.dim}, ${vars.colorMod.lightness.dim})`,
});

globalStyle(`${QueueItemMain}.${QueueStatus.error}`, {
  borderColor: `hsl(${vars.errorHue}, ${vars.colorMod.saturation.normal}, ${vars.colorMod.lightness.normal})`,
});

export const QueueButtons = style({
  display: "flex",
  flexDirection: "row",
  justifyContent: "space-between",
  alignItems: "center",
});