import React from "react";
import {
  ModifierPreview,
  useImageCreate
} from "../../../stores/imageCreateStore";

import { API_URL } from "../../../api";

import {
  ModifierTagMain,
  tagPreview
} from "./modifierTags.css";

interface ModifierTagProps {
  name: string;
  category: string;
  previews: ModifierPreview[];
}

export default function ModifierTag({ name, category, previews }: ModifierTagProps) {

  const previewType: 'portrait' | 'landscape' = "portrait";

  const hasTag = useImageCreate((state) => state.hasTag(category, name))
    ? "selected"
    : "";
  const toggleTag = useImageCreate((state) => state.toggleTag);

  const _toggleTag = () => {
    toggleTag(category, name);
  };

  return (
    <div className={[ModifierTagMain, hasTag].join(" ")} onClick={_toggleTag}>
      <p>{name}</p>
      <div className={tagPreview}>
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
      </div>
    </div>
  );
}
