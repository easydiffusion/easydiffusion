import React, { useEffect } from "react";

import {
  AppLayout,
  HeaderLayout,
  CreateLayout,
  DisplayLayout,
  FooterLayout,
} from "./home.css";

import { useQuery } from "@tanstack/react-query";
import { getSaveDirectory, loadModifications } from "../../api";
import Mockifiers from "../../components/organisms/creationPanel/imageModifiers/modifiers.mock";

import { useImageCreate } from "../../stores/imageCreateStore";

// Todo - import components here
import HeaderDisplay from "../../components/organisms/headerDisplay";
import DisplayPanel from "../../components/organisms/displayPanel";
import FooterDisplay from "../../components/organisms/footerDisplay";
import CreationTabs from "../../components/layouts/CreationTabs";

function Home() {
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
      // @ts-expect-error
      setAllModifiers(Mockifiers);
    }
  }, [setRequestOption, statusMods, dataMoads]);

  return (
    <>
      <div className={[AppLayout].join(" ")}>
        <header className={HeaderLayout}>
          <HeaderDisplay></HeaderDisplay>
        </header>
        <nav className={CreateLayout}>
          <CreationTabs></CreationTabs>
        </nav>
        <main className={DisplayLayout}>
          <DisplayPanel></DisplayPanel>
        </main>
      </div>
      <footer className={FooterLayout}>
        <FooterDisplay></FooterDisplay>
      </footer>
    </>
  );
}

export default Home;
