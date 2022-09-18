import React, { useEffect, useState } from "react";
import { useImageCreate } from "../../../../../stores/imageCreateStore";

import { useCreateUI } from "../../creationPanelUIStore";

import {
  MenuButton, //@ts-ignore
} from "../advancedsettings.css.ts";

export default function ImprovementSettings() {
  // these are conditionals that should be retired and inferred from the store
  const isUsingFaceCorrection = useImageCreate((state) =>
    state.isUsingFaceCorrection()
  );

  const isUsingUpscaling = useImageCreate((state) => state.isUsingUpscaling());

  const use_upscale = useImageCreate((state) =>
    state.getValueForRequestKey("use_upscale")
  );

  const show_only_filtered_image = useImageCreate((state) =>
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
    console.log("isUsingUpscaling", isUsingUpscaling);
    console.log("isUsingFaceCorrection", isUsingFaceCorrection);

    // if either are true we arent disabled
    if (isUsingFaceCorrection || use_upscale) {
      setIsFilteringDisabled(false);
    } else {
      setIsFilteringDisabled(true);
    }
  }, [isUsingFaceCorrection, isUsingUpscaling, setIsFilteringDisabled]);

  return (
    <div>
      <button
        type="button"
        className={MenuButton}
        onClick={toggleImprovementOpen}
      >
        <h4>Improvement Settings</h4>
      </button>
      {improvementOpen && (
        <>
          <div>
            <label>
              <input
                type="checkbox"
                checked={isUsingFaceCorrection}
                onChange={(e) => toggleUseFaceCorrection()}
              />
              Fix incorrect faces and eyes (uses GFPGAN)
            </label>
          </div>
          <div>
            <label>
              Upscale the image to 4x resolution using
              <select
                id="upscale_model"
                name="upscale_model"
                value={use_upscale}
                onChange={(e) => {
                  setRequestOption("use_upscale", e.target.value);
                }}
              >
                <option value="">No Uscaling</option>
                <option value="RealESRGAN_x4plus">RealESRGAN_x4plus</option>
                <option value="RealESRGAN_x4plus_anime_6B">
                  RealESRGAN_x4plus_anime_6B
                </option>
              </select>
            </label>
          </div>
          <div>
            <label>
              <input
                disabled={isFilteringDisabled}
                type="checkbox"
                checked={show_only_filtered_image}
                onChange={(e) =>
                  setRequestOption("show_only_filtered_image", e.target.checked)
                }
              />
              Show only filtered image
            </label>
          </div>
        </>
      )}
    </div>
  );
}
