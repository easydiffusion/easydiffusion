import React, { useState } from "react";
import { v4 as uuidv4 } from "uuid";
import {
  ModifierPreview,
  useImageCreate
} from "../../../stores/imageCreateStore";

import { API_URL } from "../../../api";

import {
  IconFont,
} from "../../../styles/shared.css";

import {
  ModifierTagMain,
  ModifierActions,
  tagPreview,
  TagText,
  TagToggle,
} from "./modifierTags.css";

interface ModifierTagProps {
  name: string;
  category: string;
  previews: ModifierPreview[];
}

export default function ModifierTag({ name, category, previews }: ModifierTagProps) {

  const previewType: 'portrait' | 'landscape' = "portrait";

  const [showActions, setShowActions] = useState(false);

  const handleHover = () => {
    setShowActions(true);
  };

  const handleLeave = () => {
    setShowActions(false);
  };

  const addCreateTag = useImageCreate((state) => state.addCreateTag);
  const setPositivePrompt = () => {
    addCreateTag({ id: uuidv4(), name, type: 'positive' });
  }
  const setNegativePrompt = () => {
    addCreateTag({ id: uuidv4(), name, type: 'negative' });
  }


  const hasTag = useImageCreate((state) => state.hasTag(category, name))
    ? "selected"
    : "";
  const toggleTag = useImageCreate((state) => state.toggleTag);

  const _toggleTag = () => {
    toggleTag(category, name);
  };

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
      {/* <div className={tagPreview}>
        {previews.map((preview) => {
          if (preview.name !== previewType) {
            return null;
          }
          return (
            <img
              key={preview.name}
              src={`${API_URL}/media/modifier-thumbnails/${preview.path}`}
              alt={preview.name}
              title={preview.name}
            />
          );
        })}
      </div> */}
    </div>
  );
}
