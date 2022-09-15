import React, { useEffect, useState } from "react";

import {
  AppLayout,
  HeaderLayout,
  CreateLayout,
  DisplayLayout,
  FooterLayout, // @ts-ignore
} from "./home.css.ts";

import { useQuery } from "@tanstack/react-query";
import { getSaveDirectory } from "../../../api";
import { useImageCreate } from "../../../store/imageCreateStore";

// Todo - import components here
import HeaderDisplay from "../../organisms/headerDisplay";
import CreationPanel from "../../organisms/creationPanel";
import DisplayPanel from "../../organisms/displayPanel";
import FooterDisplay from "../../organisms/footerDisplay";

function Editor() {
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

export default Editor;
