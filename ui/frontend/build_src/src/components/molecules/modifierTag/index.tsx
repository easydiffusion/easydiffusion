import React, { useState } from "react";
import { v4 as uuidv4 } from "uuid";
import {
  ModifierPreview,
  useImageCreate
} from "@stores/imageCreateStore";

import {
  IconFont,
} from "@styles/shared.css";

import {
  ModifierTagMain,
  ModifierActions,
  TagText,
  TagToggle,
} from "./modifierTags.css";

interface ModifierTagProps {
  name: string;
  category: string;
  previews: ModifierPreview[];
}

export default function ModifierTag({ name, category, previews }: ModifierTagProps) {

  // const previewType: 'portrait' | 'landscape' = "portrait";

  const [showActions, setShowActions] = useState(false);

  const handleHover = () => {
    setShowActions(true);
  };

  const handleLeave = () => {
    setShowActions(false);
  };

  const modifyPrompt = useImageCreate((state) => state.modifyPrompt);

  const setPositivePrompt = () => {
    modifyPrompt(name, 'positive');
  }
  const setNegativePrompt = () => {
    modifyPrompt(name, 'negative');
  }

  // const hasTag = useImageCreate((state) => state.hasTag(category, name))
  //   ? "selected"
  //   : "";
  // const toggleTag = useImageCreate((state) => state.toggleTag);

  // const _toggleTag = () => {
  //   toggleTag(category, name);
  // };

  // , hasTag].join(" ")
  return (
    <div className={ModifierTagMain}
      onMouseEnter={handleHover}
      onMouseLeave={handleLeave}>
      <p className={!showActions ? TagText : TagToggle}>{name}</p>
      {showActions && (
        <div className={ModifierActions}>
          <button onClick={setPositivePrompt}>
            <i className={[IconFont, 'fa-solid', 'fa-plus'].join(" ")}></i>
          </button>
          <button onClick={setNegativePrompt}>
            <i className={[IconFont, 'fa-solid', 'fa-minus'].join(" ")}></i>
          </button>
        </div>
      )}

    </div>
  );
}
