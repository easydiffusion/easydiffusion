import React from "react";

import { useImageCreate } from "../../../../../stores/imageCreateStore";
import ModifierTag from "../../../../molecules/modifierTag";

export default function ActiveTags() {
  const selectedtags = useImageCreate((state) => state.selectedTags());

  return (
    <div className="selected-tags">
      <p>Active Tags</p>
      <ul>
        {selectedtags.map((tag) => (
          <li key={tag.modifier}>
            {/* eslint-disable-next-line @typescript-eslint/no-non-null-assertion */}
            <ModifierTag category={tag.category!} name={tag.modifier} previews={tag.previews} />
          </li>
        ))}
      </ul>
    </div>
  );
}
