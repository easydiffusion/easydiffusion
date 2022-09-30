import React from "react";
import { ReactLocation, Router } from "@tanstack/react-location";
import Home from "./pages/Home";
import Settings from "./pages/Settings";
import "./Translation/config";
const location = new ReactLocation();

function App() {


  return (
    <Router
      location={location}
      routes={[
        { path: "/", element: <Home /> },
        { path: "/settings", element: <Settings /> },
      ]}
    ></Router>
  );
}

export default App;
