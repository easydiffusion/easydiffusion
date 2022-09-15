import React, { useState } from "react";
import { useImageCreate } from "../../../../../store/imageCreateStore";

import { MenuButton } 
from //@ts-ignore
'../advancedsettings.css.ts';


export default function WorkflowSettings() {
  const num_outputs = useImageCreate((state) =>
    state.getValueForRequestKey("num_outputs")
  );
  const parallelCount = useImageCreate((state) => state.parallelCount);
  const isUseAutoSave = useImageCreate((state) => state.isUseAutoSave());
  const save_to_disk_path = useImageCreate((state) =>
    state.getValueForRequestKey("save_to_disk_path")
  );
  const isSoundEnabled = useImageCreate((state) => state.isSoundEnabled());

  const setRequestOption = useImageCreate((state) => state.setRequestOptions);
  const setParallelCount = useImageCreate((state) => state.setParallelCount);
  const toggleUseAutoSave = useImageCreate((state) => state.toggleUseAutoSave);
  const toggleSoundEnabled = useImageCreate(
    (state) => state.toggleSoundEnabled
  );

  const [workflowOpen, setWorkflowOpen] = useState(true);

  const toggleWorkflowOpen = () => {
    setWorkflowOpen(!workflowOpen);
  };

  return (
    <div>
      <button
        type="button"
        className={MenuButton}
        onClick={toggleWorkflowOpen}
      >
        <h4>Workflow Settings</h4>
      </button>
      {workflowOpen && (
        <>
          <div>
            <label>
              Number of images to make:{" "}
              <input
                type="number"
                value={num_outputs}
                onChange={(e) =>
                  setRequestOption("num_outputs", parseInt(e.target.value, 10))
                }
                size={4}
              />
            </label>
          </div>
          <div>
            <label>
              Generate in parallel:
              <input
                type="number"
                value={parallelCount}
                onChange={(e) => setParallelCount(parseInt(e.target.value, 10))}
                size={4}
              />
            </label>
          </div>
          <div>
            <label>
              <input
                checked={isUseAutoSave}
                onChange={(e) => toggleUseAutoSave()}
                type="checkbox"
              />
              Automatically save to{" "}
            </label>
            <label>
              <input
                value={save_to_disk_path}
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
          </div>
          <div>
            <label>
              <input
                checked={isSoundEnabled}
                onChange={(e) => toggleSoundEnabled()}
                type="checkbox"
              />
              Play sound on task completion
            </label>
          </div>
        </>
      )}
    </div>
  );
}
