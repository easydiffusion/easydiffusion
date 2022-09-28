import React from "react";
import { useImageCreate } from "../../../../../stores/imageCreateStore";

import { useCreateUI } from "../../creationPanelUIStore";

import {
  SettingItem
} from "../../../../../styles/shared.css";

import {
  MenuButton, // @ts-expect-error
} from "../advancedsettings.css.ts";

import { useTranslation } from "react-i18next";

export default function WorkflowSettings() {
  const { t } = useTranslation();

  const numOutputs = useImageCreate((state) =>
    state.getValueForRequestKey("num_outputs")
  );
  const parallelCount = useImageCreate((state) => state.parallelCount);
  // const isUseAutoSave = useImageCreate((state) => state.isUseAutoSave());
  // const diskPath = useImageCreate((state) =>
  //   state.getValueForRequestKey("save_to_disk_path")
  // );
  // const isSoundEnabled = useImageCreate((state) => state.isSoundEnabled());

  const setRequestOption = useImageCreate((state) => state.setRequestOptions);
  const setParallelCount = useImageCreate((state) => state.setParallelCount);
  const shouldStreamImages = useImageCreate((state) => state.getValueForRequestKey("stream_image_progress"));
  // const toggleUseAutoSave = useImageCreate((state) => state.toggleUseAutoSave);


  // const toggleSoundEnabled = useImageCreate(
  //   (state) => state.toggleSoundEnabled
  // );

  const workflowOpen = useCreateUI((state) => state.isOpenAdvWorkflowSettings);
  const toggleWorkflowOpen = useCreateUI(
    (state) => state.toggleAdvWorkflowSettings
  );

  return (
    <div>
      <button type="button" className={MenuButton} onClick={toggleWorkflowOpen}>
        <h4>Workflow Settings</h4>
      </button>
      {workflowOpen && (
        <>
          <div className={SettingItem}>
            <label>
              {t("settings.amount-of-img")}{" "}
              <input
                type="number"
                value={numOutputs}
                onChange={(e) =>
                  setRequestOption("num_outputs", parseInt(e.target.value, 10))
                }
                size={4}
              />
            </label>
          </div>
          <div className={SettingItem}>
            <label>
              {t("settings.how-many")}
              <input
                type="number"
                value={parallelCount}
                onChange={(e) => setParallelCount(parseInt(e.target.value, 10))}
                size={4}
              />
            </label>
          </div>

          <div className={SettingItem}>
            <label>
              {t("settings.stream-img")}
              <input
                type="checkbox"
                checked={shouldStreamImages}
                onChange={(e) =>
                  setRequestOption("stream_image_progress", e.target.checked)
                }
              />
            </label>
          </div>
        </>
      )}
    </div>
  );
}
