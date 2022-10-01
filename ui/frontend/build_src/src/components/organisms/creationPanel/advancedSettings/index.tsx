import React, { useEffect } from "react";
import { useCreateUI } from "../creationPanelUIStore";



import {
  card
} from '../../../_recipes/card.css';

import {
  buttonStyle,
} from "../../../_recipes/button.css";

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
    <div className={card(
      {
        level: 1,
        backing: 'normal'
      }
    )}>
      <button
        type="button"
        onClick={toggleAdvancedSettingsIsOpen}
        className={buttonStyle({
          type: 'clear',
          size: 'large'
        })}
      >
        Advanced Settings
      </button>
      {advancedSettingsIsOpen && <SettingsList />}
    </div >
  );
}
