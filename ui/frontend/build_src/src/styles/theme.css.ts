import {
  createGlobalThemeContract,
  createGlobalTheme,
  createThemeContract,
  createTheme,
} from "@vanilla-extract/css";

const colors = createThemeContract({
  primary: null,
  secondary: null,
  background: null,
  warning: "yellow",
  error: "red",
  success: "green",
  text: {
    normal: null,
    dimmed: null,
  },
});

const app = createGlobalTheme(":root", {
  spacing: {
    small: "5px",
    medium: "10px",
    large: "25px",
  },
  fonts: {
    heading: "Georgia, Times, Times New Roman, serif",
    body: "system-ui",
  },
  colors: colors,
});

export const lightTheme = createTheme(colors, {
  primary: "#1E40AF",
  secondary: "#DB2777",
  background: "#EFF6FF",
  warning: "yellow",
  error: "red",
  success: "green",
  text: {
    normal: "#1F2937",
    dimmed: "#6B7280",
  },
});

export const darkTheme = createTheme(colors, {
  primary: "#60A5FA",
  secondary: "#F472B6",
  background: "rgb(32, 33, 36)",
  warning: "yellow",
  error: "red",
  success: "green",
  text: {
    normal: "#F9FAFB",
    dimmed: "#D1D5DB",
  },
});

export const vars = { ...app, colors };
