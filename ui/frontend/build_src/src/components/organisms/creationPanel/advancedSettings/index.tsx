import React, { useEffect } from "react";
import { useCreateUI } from "../creationPanelUIStore";

import { PanelBox } from "../../../../styles/shared.css";

import {
  AdvancedSettingsList,
  AdvancedSettingGrouping,
} from "./advancedsettings.css";

import ImprovementSettings from "./improvementSettings";
import PropertySettings from "./propertySettings";
import WorkflowSettings from "./workflowSettings";

function SettingsList() {
  return (
    <ul className={AdvancedSettingsList}>
      <li className={AdvancedSettingGrouping}>
        <ImprovementSettings />
      </li>
      <li className={AdvancedSettingGrouping}>
        <PropertySettings />
      </li>
      <li className={AdvancedSettingGrouping}>
        <WorkflowSettings />
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
