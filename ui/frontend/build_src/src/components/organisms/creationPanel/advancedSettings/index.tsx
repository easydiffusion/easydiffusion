import React, { useEffect } from "react";
import { useCreateUI } from "../creationPanelUIStore";

// @ts-expect-error
import { PanelBox } from "../../../../styles/shared.css.ts";

import {
  AdvancedSettingsList,
  AdvancedSettingItem, // @ts-expect-error
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
    <div className={PanelBox}>
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
