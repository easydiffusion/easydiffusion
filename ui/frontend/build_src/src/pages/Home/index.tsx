import React, { useEffect } from "react";

import {
  AppLayout,
  HeaderLayout,
  CreateLayout,
  DisplayLayout,
  FooterLayout,
} from "./home.css";

import { useQuery } from "@tanstack/react-query";
import { getSaveDirectory, loadModifications } from "@api/index";
import { useImageCreate } from "@stores/imageCreateStore";
import Mockifiers from "@organisms/creationPanel/imageModifiers/modifiers.mock";


// Todo - import components here
import HeaderDisplay from "@organisms/headerDisplay";
import BasicDisplay from "@layouts/basicDisplay";
import FooterDisplay from "@organisms/footerDisplay";
import CreationTabs from "@layouts/creationTabs";

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
          <BasicDisplay></BasicDisplay>
        </main>
      </div>
      <footer className={FooterLayout}>
        <FooterDisplay></FooterDisplay>
      </footer>
    </>
  );
}

export default Home;
