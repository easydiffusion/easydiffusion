import React, { useEffect, useState } from "react";
import { useImageCreate } from "../../../../../stores/imageCreateStore";

import { useCreateUI } from "../../creationPanelUIStore";

import {
  SettingItem,
} from "../../../../../styles/shared.css";

import {
  buttonStyle,
} from "../../../../_recipes/button.css";

import Checkbox from "../../../../atoms/checkbox";

import { useTranslation } from "react-i18next";

export default function ImprovementSettings() {
  const { t } = useTranslation();

  // these are conditionals that should be retired and inferred from the store
  const isUsingFaceCorrection = useImageCreate((state) =>
    state.isUsingFaceCorrection()
  );

  const isUsingUpscaling = useImageCreate((state) => state.isUsingUpscaling());

  const useUpscale = useImageCreate((state) =>
    state.getValueForRequestKey("use_upscale")
  );

  const filteredOnly = useImageCreate((state) =>
    state.getValueForRequestKey("show_only_filtered_image")
  );

  const toggleUseFaceCorrection = useImageCreate(
    (state) => state.toggleUseFaceCorrection
  );

  const setRequestOption = useImageCreate((state) => state.setRequestOptions);

  const improvementOpen = useCreateUI(
    (state) => state.isOpenAdvImprovementSettings
  );

  const toggleImprovementOpen = useCreateUI(
    (state) => state.toggleAdvImprovementSettings
  );

  const [isFilteringDisabled, setIsFilteringDisabled] = useState(false);
  // should probably be a store selector
  useEffect(() => {
    // if either are true we arent disabled
    if (isUsingFaceCorrection || useUpscale != "") {
      setIsFilteringDisabled(false);
    } else {
      setIsFilteringDisabled(true);
    }
  }, [isUsingFaceCorrection, isUsingUpscaling, setIsFilteringDisabled]);

  return (
    <div>
      <button
        type="button"
        className={buttonStyle({
          type: 'action',
          color: 'accent',
        })}
        onClick={toggleImprovementOpen}
      >
        Improvement Settings
      </button>
      {improvementOpen && (
        <>
          <div className={SettingItem}>
            <Checkbox
              label="Face Fixes (uses GFPGAN)"
              isChecked={isUsingFaceCorrection}
              toggleCheck={toggleUseFaceCorrection}
            ></Checkbox>
          </div>
          <div className={SettingItem}>
            <label>
              {t("settings.ups")}
              <select
                id="upscale_model"
                name="upscale_model"
                value={useUpscale}
                onChange={(e) => {
                  setRequestOption("use_upscale", e.target.value);
                }}
              >
                <option value="">{t("settings.no-ups")}</option>
                <option value="RealESRGAN_x4plus">RealESRGAN_x4plus</option>
                <option value="RealESRGAN_x4plus_anime_6B">
                  RealESRGAN_x4plus_anime_6B
                </option>
              </select>
            </label>
          </div>
          <div className={SettingItem}>
            <Checkbox
              disabled={isFilteringDisabled}
              label={t("settings.corrected")}
              isChecked={filteredOnly}
              toggleCheck={(value) => {
                setRequestOption("show_only_filtered_image", value)
              }}
            ></Checkbox>

          </div>
        </>
      )}
    </div>
  );
}
