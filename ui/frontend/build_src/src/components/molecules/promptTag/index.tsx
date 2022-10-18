import React, { useState } from "react";

import { useImageCreate } from "@stores/imageCreateStore";
import { usePromptMatrix } from "@stores/promptMatrixStore";

import {
  IconFont,
} from "@styles/shared.css";


import {
  PromptTagMain,
  TagToggle,
  TagRemoveButton,
  PromptTagText,
  PromptTagToggle
} from "./promptTag.css";

interface PromptTagProps {
  id: string;
  name: string;
  queueId?: string;
  category?: string;
  previews?: string[];
  type: string;
};

export default function PromptTag({ id, name, queueId, category, previews, type }: PromptTagProps) {

  const [showToggle, setShowToggle] = useState(false);
  const togglePromptType = usePromptMatrix((state) => state.togglePromptType);
  const removePromptModifier = usePromptMatrix((state) => state.removePromptModifier);

  const handleHover = () => {
    setShowToggle(true);
  };

  const handleLeave = () => {
    setShowToggle(false);
  };

  const handleToggle = () => {
    if (void 0 != queueId) {
      togglePromptType(queueId, id);
    }
  };

  const handleRemove = () => {
    debugger;
    if (void 0 != queueId) {
      removePromptModifier(queueId, id);
    }
  };

  return (
    <div
      onMouseEnter={handleHover}
      onMouseLeave={handleLeave}
      className={[PromptTagMain, type].join(' ')}>
      <p className={!showToggle ? PromptTagText : PromptTagToggle}>{name}</p>
      {showToggle && <button className={TagToggle} onClick={handleToggle}>
        {type === 'positive' ? <i className={[IconFont, 'fa-solid', 'fa-minus'].join(" ")}></i> : <i className={[IconFont, 'fa-solid', 'fa-plus'].join(" ")}></i>}
      </button>}
      {showToggle && <button className={TagRemoveButton} onClick={handleRemove}>
        <i className={[IconFont, 'fa-solid', 'fa-close'].join(" ")}></i>
      </button>}
    </div>
  );
};