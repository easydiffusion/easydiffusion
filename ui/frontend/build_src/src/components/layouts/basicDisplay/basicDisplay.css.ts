import { style, globalStyle } from "@vanilla-extract/css";
import { vars } from "../../../styles/theme/index.css";

import {
  tabStyles
} from "../../_recipes/tabs_headless.css";

export const basicDisplayLayout = style({
  width: "100%",
  height: "100%",
  display: "grid",
  gridTemplateColumns: "1fr 250px",
  gridTemplateRows: "minmax(0, 1fr) 250px",
  gridTemplateAreas: `
    "content info"
    "history history"`,

  overflow: "hidden",

  selectors: {
    '&[data-hide-info]': {
      gridTemplateColumns: "1fr 0px",
      gridTemplateRows: "1fr 250px",
      // gridTemplateAreas: `
      //   "content"
      //   "history"`,
    },
    '&[data-hide-history]': {
      gridTemplateColumns: "1fr 250px",
      gridTemplateRows: "1fr 0px",
      // gridTemplateAreas: `
      //   "content info"`,
    },
    '&[data-hide-info][data-hide-history]': {
      gridTemplateColumns: "1fr 0px",
      gridTemplateRows: "1fr 0px",
      // gridTemplateAreas: `
      //   "content"`,
    },
  },

  // "@media": {
  //   "screen and (max-width: 800px)": {
  //     gridTemplateColumns: "1fr",
  //     gridTemplateRows: "100px 300px 1fr",
  //     gridTemplateAreas: `
  //       "header"
  //       "create"
  //       "display"
  //     `,
  //   },
  // },

});

// globalStyle(`${basicDisplayLayout}.hideHistory`, {

// });

export const contentLayout = style({
  gridArea: "content",
});

export const infoLayout = style({
  gridArea: "info",
  position: "relative",
});


export const infoTab = style([tabStyles({}), {
  position: 'absolute',
  whiteSpace: 'nowrap',
  top: '0',
  right: '100%',
  textAlign: 'right',
  // -webkit-transform-origin: 100% 100%;
  // -webkit-transform: rotate(-90deg);
  transformOrigin: `100% 100%`,
  transform: `rotate(-90deg)`,


}]);

export const historyLayout = style({
  gridArea: "history",
  position: "relative",
});

globalStyle(`${historyLayout} > button`, {
  position: "absolute",
  top: '-29px',
});

export const displayContainer = style({
  flexGrow: 1,
  overflow: 'auto',
  display: "flex",
});

export const displayData = style({
  width: '250px',
  height: '100%',
  backgroundColor: 'rebeccapurple',
  position: 'relative',
});

export const DataTab = style([tabStyles(), {
  position: 'absolute',
  top: '0',
  left: '0',

  // pretty sure this is a magic number
  transformOrigin: '37% 275%',
  transform: 'rotate(-90deg)',
}]);


// export const previousImages = style({
//   minHeight: '250px',
// });
