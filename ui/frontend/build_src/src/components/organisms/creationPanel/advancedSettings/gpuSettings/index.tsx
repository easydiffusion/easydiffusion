import React from "react";
import { useImageCreate } from "../../../../../stores/imageCreateStore";

import { useCreateUI } from "../../creationPanelUIStore";

import {
  SettingItem, // @ts-expect-error
} from "../../../../../styles/shared.css.ts";

import {
  MenuButton, // @ts-expect-error
} from "../advancedsettings.css.ts";
import {useTranslation} from "react-i18next";

export default function GpuSettings() {
  const { t } = useTranslation();

  const turbo = useImageCreate((state) => state.getValueForRequestKey("turbo"));
  const useCpu = useImageCreate((state) =>
    state.getValueForRequestKey("use_cpu")
  );
  const useFullPrecision = useImageCreate((state) =>
    state.getValueForRequestKey("use_full_precision")
  );

  const setRequestOption = useImageCreate((state) => state.setRequestOptions);

  const gpuOpen = useCreateUI((state) => state.isOpenAdvGPUSettings);
  const toggleGpuOpen = useCreateUI((state) => state.toggleAdvGPUSettings);

  return (
    <div>
      <button type="button" className={MenuButton} onClick={toggleGpuOpen}>
        <h4>GPU Settings</h4>
      </button>
      {gpuOpen && (
        <>
          <div className={SettingItem}>
            <label>
              <input
                checked={turbo}
                onChange={(e) => setRequestOption("turbo", e.target.checked)}
                type="checkbox"
              />
              {t("advanced-settings.turbo")} {t("advanced-settings.turbo-disc")}
            </label>
          </div>
          <div className={SettingItem}>
            <label>
              <input
                type="checkbox"
                checked={useCpu}
                onChange={(e) => setRequestOption("use_cpu", e.target.checked)}
              />
              {t("advanced-settings.cpu")} {t("advanced-settings.cpu-disc")}
            </label>
          </div>
          <div className={SettingItem}>
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
          </div>
        </>
      )}
    </div>
  );
}
