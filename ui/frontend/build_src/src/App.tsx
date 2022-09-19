import React, { useState } from "react";
import { ReactLocation, Router } from "@tanstack/react-location";
import Home from "./pages/Home";
import Settings from "./pages/Settings";
// @ts-ignore
import { darkTheme, lightTheme } from "./styles/theme/index.css.ts";
import './Translation/config';
const location = new ReactLocation();

function App() {
  // just check for the theme one 1 time

  // var { matches } = window.matchMedia('(prefers-color-scheme: dark)')
  const matches = true;
  const themeClass = matches ? darkTheme : lightTheme;

  return (
    <Router
      location={location}
      routes={[
        { path: "/", element: <Home className={themeClass} /> },
        { path: "/settings", element: <Settings className={themeClass} /> },
      ]}
    >
    </Router>
  );
}

export default App;
