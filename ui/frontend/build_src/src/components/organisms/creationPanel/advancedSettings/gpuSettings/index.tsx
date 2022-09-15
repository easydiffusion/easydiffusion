import React, { useState } from "react";
import { useImageCreate } from "../../../../../store/imageCreateStore";

import { MenuButton } 
from //@ts-ignore
'../advancedsettings.css.ts'


export default function GpuSettings() {
  const turbo = useImageCreate((state) => state.getValueForRequestKey("turbo"));
  const use_cpu = useImageCreate((state) =>
    state.getValueForRequestKey("use_cpu")
  );
  const use_full_precision = useImageCreate((state) =>
    state.getValueForRequestKey("use_full_precision")
  );

  const setRequestOption = useImageCreate((state) => state.setRequestOptions);

  const [gpuOpen, setGpuOpen] = useState(false);

  const toggleGpuOpen = () => {
    setGpuOpen(!gpuOpen);
  };

  return (
    <div>
      <button
        type="button"
        className={MenuButton}
        onClick={toggleGpuOpen}
      >
        <h4>GPU Settings</h4>
      </button>
      {gpuOpen && (
        <>
          <div>
            <label>
              <input
                checked={turbo}
                onChange={(e) => setRequestOption("turbo", e.target.checked)}
                type="checkbox"
              />
              Turbo mode (generates images faster, but uses an additional 1 GB
              of GPU memory)
            </label>
          </div>
          <div>
            <label>
              <input
                type="checkbox"
                checked={use_cpu}
                onChange={(e) => setRequestOption("use_cpu", e.target.checked)}
              />
              Use CPU instead of GPU (warning: this will be *very* slow)
            </label>
          </div>
          <div>
            <label>
              <input
                checked={use_full_precision}
                onChange={(e) =>
                  setRequestOption("use_full_precision", e.target.checked)
                }
                type="checkbox"
              />
              Use full precision (for GPU-only. warning: this will consume more
              VRAM)
            </label>
          </div>
        </>
      )}
    </div>
  );
}
