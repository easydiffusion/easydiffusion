import {
  createGlobalTheme,
  createThemeContract,
  createTheme,
} from "@vanilla-extract/css";

/**
 * Colors are all the same across the themes, this is just to set up a contract
 * Colors can be decided later. I am just the architect.
 * Tried to pull things from the original app. 
 * 
 * Lots of these arent used yet, but once they are defined and useable then they can be set.
 */

const colors = createThemeContract({

  brand: '##5000b9', // purple
  brandDimmed: '#5000b9', // purple
  brandHover: '5d00d6', // bringhter purple
  brandHoverDimmed: '#5d00d6', // bringhter purple
  brandActive: '#5d00d6', // bringhter purple
  brandActiveDimmed: '#5d00d6', // bringhter purple
  brandAccent: '#28004e', // darker purple
  brandAccentDimmed: '#28004e', // darker purple
  brandAccentActive: '#28004e', // darker purple

  secondary: '#0b8334', // green
  secondaryDimmed: '#0b8334', // green
  secondaryHover: '#0b8334', // green
  secondaryHoverDimmed: '#0b8334', // green
  secondaryActive: '#0b8334', // green
  secondaryActiveDimmed: '#0b8334', // green
  secondaryAccent: '#0b8334', // green
  secondaryAccentDimmed: '#0b8334', // green
  secondaryAccentActive: '#0b8334', // green

  background: '#202124', // dark grey
  backgroundAccent: ' #383838', // lighter grey
  backgroundAlt: '#2c2d30', // med grey
  backgroundAltAccent: '#383838', // lighter grey

  text: {
    normal: '#ffffff', // white
    dimmed: '#d1d5db', // off white

    secondary: '#e7ba71', // orange
    secondaryDimmed: '#7d6641', // muted orange
  },

  warning: "yellow",
  error: "red",
  success: "green",

});

const app = createGlobalTheme(":root", {
  spacing: {
    small: "5px",
    medium: "10px",
    large: "25px",
  },
  fonts: {
    body: "Arial, Helvetica, sans-serif;",
  },
  colors: colors,
});


export const lightTheme = createTheme(colors, {
  brand: '##5000b9', // purple
  brandDimmed: '#5000b9', // purple
  brandHover: '5d00d6', // bringhter purple
  brandHoverDimmed: '#5d00d6', // bringhter purple
  brandActive: '#5d00d6', // bringhter purple
  brandActiveDimmed: '#5d00d6', // bringhter purple
  brandAccent: '#28004e', // darker purple
  brandAccentDimmed: '#28004e', // darker purple
  brandAccentActive: '#28004e', // darker purple

  secondary: '#0b8334', // green
  secondaryDimmed: '#0b8334', // green
  secondaryHover: '#0b8334', // green
  secondaryHoverDimmed: '#0b8334', // green
  secondaryActive: '#0b8334', // green
  secondaryActiveDimmed: '#0b8334', // green
  secondaryAccent: '#0b8334', // green
  secondaryAccentDimmed: '#0b8334', // green
  secondaryAccentActive: '#0b8334', // green

  background: '#202124', // dark grey
  backgroundAccent: ' #383838', // lighter grey
  backgroundAlt: '#2c2d30', // med grey
  backgroundAltAccent: '#383838', // lighter grey

  text: {
    normal: '#ffffff', // white
    dimmed: '#d1d5db', // off white

    secondary: '#e7ba71', // orange
    secondaryDimmed: '#7d6641', // muted orange
  },

  warning: "yellow",
  error: "red",
  success: "green",

});


export const darkTheme = createTheme(colors, {
  brand: '##5000b9', // purple
  brandDimmed: '#5000b9', // purple
  brandHover: '5d00d6', // bringhter purple
  brandHoverDimmed: '#5d00d6', // bringhter purple
  brandActive: '#5d00d6', // bringhter purple
  brandActiveDimmed: '#5d00d6', // bringhter purple
  brandAccent: '#28004e', // darker purple
  brandAccentDimmed: '#28004e', // darker purple
  brandAccentActive: '#28004e', // darker purple

  secondary: '#0b8334', // green
  secondaryDimmed: '#0b8334', // green
  secondaryHover: '#0b8334', // green
  secondaryHoverDimmed: '#0b8334', // green
  secondaryActive: '#0b8334', // green
  secondaryActiveDimmed: '#0b8334', // green
  secondaryAccent: '#0b8334', // green
  secondaryAccentDimmed: '#0b8334', // green
  secondaryAccentActive: '#0b8334', // green

  background: '#202124', // dark grey
  backgroundAccent: ' #383838', // lighter grey
  backgroundAlt: '#2c2d30', // med grey
  backgroundAltAccent: '#383838', // lighter grey

  text: {
    normal: '#ffffff', // white
    dimmed: '#d1d5db', // off white

    secondary: '#e7ba71', // orange
    secondaryDimmed: '#7d6641', // muted orange
  },

  warning: "yellow",
  error: "red",
  success: "green",

});




export const vars = { ...app, colors };
