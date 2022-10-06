import { style, globalStyle } from '@vanilla-extract/css'
import { recipe } from "@vanilla-extract/recipes";


export const progressImageDisplayStyle = recipe({
  base: {
    display: "flex",
    width: "100%",
    height: "100%",
  },
  variants: {
    orientation: {
      horizontal: {
        flexDirection: "row",
      },
      vertical: {
        flexDirection: "column",
      },
    },
  },
});

// this is a hack to work round a bug in vanilla-extract
export const progressImage = recipe({
  base: {
    position: "relative",
    objectFit: "contain",
    width: "100%",
    height: "100%",
  },
  variants: {
    orientation: {
      horizontal: {
        width: "80%",
      },
      vertical: {
        width: "20%",
      },
    },
  },
});

// this would be best but it doesn't work
// globalStyle(`${progressImageDisplayStyle(
//   { orientation: "vertical" }
// )} > img`, {
//   width: "80%",
// });
// globalStyle(`${progressImageDisplayStyle(
//   { orientation: "horizontal" }
// )} > img`, {
//   width: "100px",
// });

// This would be better but it doesn't work
// export const progressImage = style({
//   // width: "100px",

//   selectors: {
//     [`${progressImageDisplayStyle(
//       { orientation: "horizontal" }
//     )} &`]: {
//       width: "25%",
//     },
//     [`${progressImageDisplayStyle(
//       { orientation: "vertical" }
//     )} &`]: {
//       width: "80%",
//     }
//   }

// });