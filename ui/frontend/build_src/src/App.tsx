import React, { useEffect, useState } from "react";


import { 
  AppLayout,
  HeaderLayout,
  CreateLayout,
  DisplayLayout,
  FooterLayout
} 
from './app.css.ts';

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
    <div className={AppLayout}>
      <header className={HeaderLayout}>
        <HeaderDisplay></HeaderDisplay>
      </header>
      <nav className={CreateLayout}>
        <CreationPanel></CreationPanel>
      </nav>
      <main className={DisplayLayout}>
        <DisplayPanel></DisplayPanel>
      </main>
      <footer className={FooterLayout}>
        <FooterDisplay></FooterDisplay>
      </footer>
    </div>
  );
}

export default App;
