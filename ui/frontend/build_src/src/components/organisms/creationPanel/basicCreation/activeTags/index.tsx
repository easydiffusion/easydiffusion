import React from "react";

import { useImageCreate } from "../../../../../store/imageCreateStore";
import ModifierTag from "../../../../atoms/modifierTag";

export default function ActiveTags() {
  const selectedtags = useImageCreate((state) => state.selectedTags());
  return (
    <div className="selected-tags">
    <p>Active Tags</p>
    <ul>
      {selectedtags.map((tag) => (
        <li key={tag}>
          <ModifierTag name={tag}></ModifierTag>
        </li>
      ))}
    </ul>
  </div>
  );
}