import React from "react";
import { useImageCreate } from "../../../../../stores/imageCreateStore";
import { useImageDisplay } from "../../../../../stores/imageDisplayStore";

import { useCreateUI } from "../../creationPanelUIStore";

import {
  SettingItem,
} from "../../../../../styles/shared.css";

import {
  buttonStyle,
} from "../../../../_recipes/button.css";


import Checkbox from "../../../../atoms/headlessCheckbox";
import NumberInput from "../../../../atoms/numberInput";

import ModelOptions from "./modelOptions";
// import FileOptions from "./fileOptions";


import { useTranslation } from "react-i18next";

export default function WorkflowSettings() {
  const { t } = useTranslation();

  const numOutputs = useImageCreate((state) =>
    state.getValueForRequestKey("num_outputs")
  );
  const parallelCount = useImageCreate((state) => state.parallelCount);

  const setRequestOption = useImageCreate((state) => state.setRequestOptions);
  const setParallelCount = useImageCreate((state) => state.setParallelCount);
  const shouldStreamImages = useImageCreate((state) => state.getValueForRequestKey("stream_image_progress"));

  const workflowOpen = useCreateUI((state) => state.isOpenAdvWorkflowSettings);
  const toggleWorkflowOpen = useCreateUI(
    (state) => state.toggleAdvWorkflowSettings
  );


  const shouldDisplayWhenComplete = useImageDisplay((state) => state.shouldDisplayWhenComplete);
  const toggleDisplayComplete = useImageDisplay((state) => state.toggleDisplayComplete);


  return (
    <div>
      <button type="button" className={buttonStyle({
        type: 'action',
        color: 'accent',
      })} onClick={toggleWorkflowOpen}>
        Workflow Settings
      </button>
      {workflowOpen && (
        <>
          <div className={SettingItem}>
            <NumberInput
              label={t("settings.amount-of-img")}
              value={numOutputs}
              min={1}
              onChange={(value) => setRequestOption("num_outputs", value)}
            ></NumberInput>
          </div>

          <div className={SettingItem}>
            <NumberInput
              label={t("settings.how-many")}
              min={1}
              value={parallelCount}
              onChange={(value) => setParallelCount(value)}
            ></NumberInput>
          </div>

          <div className={SettingItem}>
            <ModelOptions></ModelOptions>
          </div>

          {/* <div className={SettingItem}>
            <FileOptions></FileOptions>
          </div> */}

          <div className={SettingItem}>
            <Checkbox
              label={t("settings.stream-img")}
              isChecked={shouldStreamImages}
              toggleCheck={(value) =>
                setRequestOption("stream_image_progress", value)
              }
            ></Checkbox>
          </div>

          <div className={SettingItem}>
            <Checkbox
              isChecked={shouldDisplayWhenComplete}
              label="Display Completed"
              toggleCheck={toggleDisplayComplete}
            ></Checkbox>
          </div>


        </>
      )}
    </div>
  );
}
