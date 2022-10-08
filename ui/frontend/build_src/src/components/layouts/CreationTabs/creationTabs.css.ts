import { style, globalStyle } from "@vanilla-extract/css";
import { vars } from "../../../styles/theme/index.css";
import {
  tabPanelStyles,
} from "../../_recipes/tabs_headless.css";


export const TabpanelScrollFlop = style([tabPanelStyles(), {
  direction: 'rtl',
  // position: 'relative',
  overflow: 'overlay',
  "::-webkit-scrollbar": {
    position: 'absolute',
    width: "6px",
    backgroundColor: vars.backgroundAccentMain,
  },

  "::-webkit-scrollbar-thumb": {
    backgroundColor: vars.backgroundDark,
    borderRadius: "4px",
  },


  // "::-webkit-scrollbar-button: {
  //   backgroundColor: vars.backgroundDark,
  // }


}]);

globalStyle(`${TabpanelScrollFlop} > *`, {
  direction: 'ltr',

});
