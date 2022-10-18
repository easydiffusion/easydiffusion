import React from "react";

import HeadlessDisclosure from "../../../atoms/headlessDisclosure";

import MakeButton from "./makeButton";
import RequestCount from "./requestCount";
import PromptCreator from "./promptCreator";
import SeedImage from "./seedImage";
import ModificationPanel from "./modificationPanel";

import {
  CreationBasicMain,
} from "./basicCreation.css";


export default function BasicCreation() {
  return (
    <div className={CreationBasicMain}>

      <PromptCreator></PromptCreator>
      <SeedImage></SeedImage>
      <div>
        <MakeButton ></MakeButton>
        <RequestCount ></RequestCount>
      </div>

      <HeadlessDisclosure
        buttonText="Prompt Matrix"
      >
        <ModificationPanel></ModificationPanel>
      </HeadlessDisclosure>

    </div>
  );
}
