import React from "react";

import { useCreateUI } from "../../creationPanelUIStore";

export default function ShowQueue() {

  const showQueue = useCreateUI((state) => state.showQueue);
  const toggleQueue = useCreateUI((state) => state.toggleQueue);

  return (
    <label>
      <input
        type="checkbox"
        checked={showQueue}
        onChange={() => toggleQueue()}
      >
      </input>
      Display
    </label>
  );
}