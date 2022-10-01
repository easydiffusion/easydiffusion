import { style, globalStyle } from "@vanilla-extract/css";

import {
  tabPanelStyles,
} from "../../_recipes/tabs_headless.css";


export const TabpanelScrollFlop = style([tabPanelStyles(), {
  direction: 'rtl',
}]);

globalStyle(`${TabpanelScrollFlop} > *`, {
  direction: 'ltr',
});
