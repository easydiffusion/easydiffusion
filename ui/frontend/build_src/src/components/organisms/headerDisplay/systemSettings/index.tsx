/* eslint-disable @typescript-eslint/strict-boolean-expressions */
import React from "react";
import { Popover } from '@headlessui/react';
import { useTranslation } from "react-i18next";

import { useImageCreate } from "../../../../stores/imageCreateStore";

import BetaMode from "../../../molecules/betaMode";

import Checkbox from "../../../atoms/headlessCheckbox";


import {
  IconFont,
  SettingItem
} from "../../../../styles/shared.css";

import {
  PopoverMain,
  PopoverButtonStyle,
  PopoverPanelMain,
} from "../../../_recipes/popover_headless.css";

import {
  SettingContent
} from "./systemSettings.css";


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
              <Checkbox
                label={t("storage.ast")}
                isChecked={isUseAutoSave}
                toggleCheck={toggleUseAutoSave}
              ></Checkbox>

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
              <Checkbox
                label={t("advanced-settings.sound")}
                isChecked={isSoundEnabled}
                toggleCheck={toggleSoundEnabled}
              ></Checkbox>
            </li>
            <li className={SettingItem}>
              <Checkbox
                label={`${t("advanced-settings.turbo")} ${t("advanced-settings.turbo-disc")}`}
                isChecked={turbo}
                toggleCheck={(value) => setRequestOption("turbo", value)}
              ></Checkbox>
            </li>
            <li className={SettingItem}>
              <Checkbox
                label={`${t("advanced-settings.cpu")} ${t("advanced-settings.cpu-disc")}`}
                isChecked={useCpu}
                toggleCheck={(value) => setRequestOption("use_cpu", value)}
              ></Checkbox>
            </li>
            <li className={SettingItem}>
              <Checkbox
                label={`${t("advanced-settings.gpu")} ${t("advanced-settings.gpu-disc")}`}
                isChecked={useFullPrecision}
                toggleCheck={(value) => setRequestOption("use_full_precision", value)}
              ></Checkbox>
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
