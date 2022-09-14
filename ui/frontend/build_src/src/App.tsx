import React, { useEffect, useState } from "react";
import "./App.css";

import { useQuery } from "@tanstack/react-query";
import { getSaveDirectory } from "./api";
import { useImageCreate } from "./store/imageCreateStore";

// Todo - import components here
import HeaderDisplay from "./components/headerDisplay";
import CreationPanel from "./components/creationPanel";
import DisplayPanel from "./components/displayPanel";
import FooterDisplay from "./components/footerDisplay";

function App() {
  // Get the original save directory
  const setRequestOption = useImageCreate((state) => state.setRequestOptions);
  const { status, data } = useQuery(["SaveDir"], getSaveDirectory);
  useEffect(() => {
    if (status === "success") {
      setRequestOption("save_to_disk_path", data);
    }
  }, [setRequestOption, status, data]);

  return (
    <div className="App">
      <header className="header-layout">
        <HeaderDisplay></HeaderDisplay>
      </header>
      <nav className="create-layout">
        <CreationPanel></CreationPanel>
      </nav>
      <main className="display-layout">
        <DisplayPanel></DisplayPanel>
      </main>
      <footer className="footer-layout">
        <FooterDisplay></FooterDisplay>
      </footer>
    </div>
  );
}

export default App;
