import {
  createGlobalTheme,
  createThemeContract,
  createTheme,
} from "@vanilla-extract/css";

const app = createGlobalTheme(":root", {
  spacing: {
    none: "0",
    min: "2px",
    small: "5px",
    medium: "10px",
    large: "25px",
  },

  trim: {
    smallBorderRadius: "5px",
  },

  fonts: {
    body: "Arial, Helvetica, sans-serif;",
    // IconFont is a shared class for now
    sizes: {
      Title: "2em",
      Headline: "1.5em",
      Subheadline: "1.20em",
      SubSubheadline: "1.1em",
      Body: "1em",
      Plain: "0.8em",
      Caption: ".75em",
      Overline: ".5em",
    },
  },
  // colors,

  // 60 degree color difference
  // step downs
  brandHue: '265', // purple
  secondaryHue: '225', // deep blue
  tertiaryHue: '145', // grass green

  // step ups
  errorHue: '0',
  warningHue: '25', // orange
  successHue: '85', // green

  colorMod: {
    saturation: {
      bright: "100%",
      normal: "60%",
      dimmed: "50%",
      dim: "30%",
    },
    lightness: {
      normal: "45%",
      bright: "60%",
      dim: "40%",
    },
  },

  // is the secondary hue
  backgroundMain: 'hsl(225, 6%, 13%)',
  backgroundLight: 'hsl(225, 4%, 18%)',
  backgroundDark: 'hsl(225, 3%, 7%)',
  backgroundAccentMain: 'hsl(225, 6%, 30%)',

  backgroundAccentHue: '225',
  backgroundAccentSaturation: '26%',
  backgroundAccentLightness: '70%',

  // this is depricated
  colors: {
    text: {
      normal: "#ffffff", // white
      dimmed: "#d1d5db", // off white

      secondary: "#ffffff", // white
      secondaryDimmed: "#d1d5db", // off white

      accent: "#e7ba71", // orange
      accentDimmed: "#7d6641", // muted orange
    },
    link: "#0066cc", // blue
  }
});

export const vars = { ...app };
