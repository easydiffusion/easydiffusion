import React, { useEffect } from "react";
import { useCreateUI } from "../creationPanelUIStore";

import {
  AdvancedSettingsList,
  AdvancedSettingItem, // @ts-ignore
} from "./advancedsettings.css.ts";

import ImprovementSettings from "./improvementSettings";
import PropertySettings from "./propertySettings";
import WorkflowSettings from "./workflowSettings";
import GpuSettings from "./gpuSettings";

import BetaMode from "../../../molecules/betaMode";

function SettingsList() {
  return (
    <ul className={AdvancedSettingsList}>
      <li className={AdvancedSettingItem}>
        <ImprovementSettings />
      </li>
      <li className={AdvancedSettingItem}>
        <PropertySettings />
      </li>
      <li className={AdvancedSettingItem}>
        <WorkflowSettings />
      </li>
      <li className={AdvancedSettingItem}>
        <GpuSettings />
      </li>

      <li className={AdvancedSettingItem}>
        <BetaMode />
      </li>

    </ul>
  );
}

export default function AdvancedSettings() {
  const advancedSettingsIsOpen = useCreateUI(
    (state) => state.isOpenAdvancedSettings
  );

  const toggleAdvancedSettingsIsOpen = useCreateUI(
    (state) => state.toggleAdvancedSettings
  );

  return (
    <div className="panel-box">
      <button
        type="button"
        onClick={toggleAdvancedSettingsIsOpen}
        className="panel-box-toggle-btn"
      >
        <h3>Advanced Settings</h3>
      </button>
      {advancedSettingsIsOpen && <SettingsList />}
    </div>
  );
}
