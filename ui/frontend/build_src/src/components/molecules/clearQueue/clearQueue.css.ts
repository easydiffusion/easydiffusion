import { style } from "@vanilla-extract/css";

import { vars } from "../../../styles/theme/index.css";

import { BrandedButton } from "../../../styles/shared.css";

export const ClearQueueButton = style([BrandedButton, {
  fontSize: vars.fonts.sizes.Headline,
}]);