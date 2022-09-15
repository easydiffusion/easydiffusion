import React, { useEffect } from "react";
import { useImageCreate } from "../../../../store/imageCreateStore";
import { 
  AdvancedSettingsList,
  AdvancedSettingItem
} // @ts-ignore 
from "./advancedsettings.css.ts";

import ImprovementSettings from "./improvementSettings";
import PropertySettings from "./propertySettings";
import WorkflowSettings from "./workflowSettings";
import GpuSettings from "./gpuSettings";

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
    </ul>
  );
}

export default function AdvancedSettings() {
  const advancedSettingsIsOpen = useImageCreate(
    (state) => state.uiOptions.advancedSettingsIsOpen
  );

  const toggleAdvancedSettingsIsOpen = useImageCreate(
    (state) => state.toggleAdvancedSettingsIsOpen
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
