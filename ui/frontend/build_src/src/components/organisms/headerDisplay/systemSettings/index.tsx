/* eslint-disable @typescript-eslint/strict-boolean-expressions */
import React from "react";
import { Popover } from '@headlessui/react';
import { useTranslation } from "react-i18next";

import { useImageCreate } from "../../../../stores/imageCreateStore";

import BetaMode from "../../../molecules/betaMode";


import {
  IconFont,
  SettingItem
} from "../../../../styles/shared.css";

import {
  PopoverMain,
  PopoverButtonStyle,
  PopoverPanelMain,
} from "../../../_headless/popover/index.css";

import {
  SettingContent
} from "./systemSettings.css";

// import {
//   SwitchGroupMain,
//   SwitchMain,
//   SwitchLabel,
//   SwitchEnabled,
//   SwitchPill,
// } from "../../../_headless/switch/index.css";


export default function SystemSettings() {
  const { t } = useTranslation();

  const isUseAutoSave = useImageCreate((state) => state.isUseAutoSave());
  const diskPath = useImageCreate((state) =>
    state.getValueForRequestKey("save_to_disk_path")
  );

  const turbo = useImageCreate((state) => state.getValueForRequestKey("turbo"));
  const useCpu = useImageCreate((state) =>
    state.getValueForRequestKey("use_cpu")
  );
  const useFullPrecision = useImageCreate((state) =>
    state.getValueForRequestKey("use_full_precision")
  );

  const isSoundEnabled = useImageCreate((state) => state.isSoundEnabled());

  const setRequestOption = useImageCreate((state) => state.setRequestOptions);
  const toggleUseAutoSave = useImageCreate((state) => state.toggleUseAutoSave);
  const toggleSoundEnabled = useImageCreate(
    (state) => state.toggleSoundEnabled
  );

  return (
    <Popover className={PopoverMain}>
      <Popover.Button className={PopoverButtonStyle}>
        <i className={[IconFont, 'fa-solid', 'fa-gear'].join(" ")}></i>
        Settings
      </Popover.Button>

      <Popover.Panel className={PopoverPanelMain}>
        <div className={SettingContent}>
          <h4>System Settings</h4>
          <ul>
            <li className={SettingItem}>
              <label>
                <input
                  checked={isUseAutoSave}
                  onChange={(e) => toggleUseAutoSave()}
                  type="checkbox"
                />
                {t("storage.ast")}{" "}
              </label>
              <label>
                <input
                  value={diskPath}
                  onChange={(e) =>
                    setRequestOption("save_to_disk_path", e.target.value)
                  }
                  size={40}
                  disabled={!isUseAutoSave}
                />
                <span className="visually-hidden">
                  Path on disk where images will be saved
                </span>
              </label>
            </li>
            <li className={SettingItem}>
              <label>
                <input
                  checked={isSoundEnabled}
                  onChange={(e) => toggleSoundEnabled()}
                  type="checkbox"
                />
                {t("advanced-settings.sound")}
              </label>


              {/* <Switch.Group>
                <Switch.Label passive> <>{t("advanced-settings.sound")}</> </Switch.Label>
                <Switch checked={isSoundEnabled} onChange={toggleSoundEnabled} className={SwitchMain}>
                  <span
                    className={SwitchPill}
                  />
                </Switch>
              </Switch.Group> */}
            </li>


            <li className={SettingItem}>
              <label>
                <input
                  checked={turbo}
                  onChange={(e) => setRequestOption("turbo", e.target.checked)}
                  type="checkbox"
                />
                {t("advanced-settings.turbo")} {t("advanced-settings.turbo-disc")}
              </label>
            </li>
            <li className={SettingItem}>
              <label>
                <input
                  type="checkbox"
                  checked={useCpu}
                  onChange={(e) => setRequestOption("use_cpu", e.target.checked)}
                />
                {t("advanced-settings.cpu")} {t("advanced-settings.cpu-disc")}
              </label>
            </li>
            <li className={SettingItem}>
              <label>
                <input
                  checked={useFullPrecision}
                  onChange={(e) =>
                    setRequestOption("use_full_precision", e.target.checked)
                  }
                  type="checkbox"
                />
                {t("advanced-settings.gpu")} {t("advanced-settings.gpu-disc")}
              </label>
            </li>

            <li className={SettingItem}>
              <BetaMode />
            </li>





          </ul>
        </div>
      </Popover.Panel>
    </Popover>

  );
}
