import React, { useEffect, useState } from "react";

import {
  AppLayout,
  HeaderLayout,
  CreateLayout,
  DisplayLayout,
  FooterLayout, // @ts-ignore
} from "./home.css.ts";

import { useQuery } from "@tanstack/react-query";
import { getSaveDirectory, loadModifications } from "../../api";
import Mockifiers from "../../components/organisms/creationPanel/imageModifiers/modifiers.mock";

import { useImageCreate } from "../../stores/imageCreateStore";

// Todo - import components here
import HeaderDisplay from "../../components/organisms/headerDisplay";
import CreationPanel from "../../components/organisms/creationPanel";
import DisplayPanel from "../../components/organisms/displayPanel";
import FooterDisplay from "../../components/organisms/footerDisplay";

function Home({ className }: { className: any }) {
  // Get the original save directory
  const setRequestOption = useImageCreate((state) => state.setRequestOptions);

  const { status: statusSave, data: dataSave } = useQuery(
    ["SaveDir"],
    getSaveDirectory
  );
  const { status: statusMods, data: dataMoads } = useQuery(
    ["modifications"],
    loadModifications
  );

  const setAllModifiers = useImageCreate((state) => state.setAllModifiers);

  useEffect(() => {
    if (statusSave === "success") {
      setRequestOption("save_to_disk_path", dataSave);
    }
  }, [setRequestOption, statusSave, dataSave]);

  useEffect(() => {
    if (statusMods === "success") {
      setAllModifiers(dataMoads);
    } else if (statusMods === "error") {
      // @ts-ignore
      setAllModifiers(Mockifiers);
    }
  }, [setRequestOption, statusMods, dataMoads]);

  return (
    <div className={[AppLayout, className].join(" ")}>
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

export default Home;
