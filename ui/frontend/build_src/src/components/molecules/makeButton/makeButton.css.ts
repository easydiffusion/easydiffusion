import { style } from "@vanilla-extract/css";

import { vars } from "../../../styles/theme/index.css";

import { BrandedButton } from "../../../styles/shared.css";

export const MakeButtonStyle = style([BrandedButton, {
  width: "100%",
  fontSize: vars.fonts.sizes.Headline,
}]);
