import { recipe } from "@vanilla-extract/recipes";

export const button = recipe({
  variants: {
    color: {
      neutral: { background: "whitesmoke" },
      brand: { background: "blueviolet" },
      accent: { background: "slateblue" },
    },
    size: {
      small: { padding: 12 },
      medium: { padding: 16 },
      large: { padding: 24 },
    },
  },
});

// export const card = recipe({
//   variants: {
//     color: {

//       alt: { background: 'whitesmoke' },
