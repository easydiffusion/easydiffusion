import React, { useState } from "react";

import { useImageCreate } from "../../../stores/imageCreateStore";

import {
  IconFont,
} from "../../../styles/shared.css";

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
  category?: string;
  previews?: string[];
  type: string;
};

export default function PromptTag({ id, name, category, previews, type }: PromptTagProps) {

  const [showToggle, setShowToggle] = useState(false);

  const removeCreateTag = useImageCreate((state) => state.removeCreateTag);
  const changeCreateTagType = useImageCreate((state) => state.changeCreateTagType);

  const handleHover = () => {
    setShowToggle(true);
  };

  const handleLeave = () => {
    setShowToggle(false);
  };

  const toggleType = () => {
    if (type === 'positive') {
      changeCreateTagType(id, 'negative');
    }
    else {
      changeCreateTagType(id, 'positive');
    }
  };

  const handleRemove = () => {
    console.log('remove');
    removeCreateTag(id);
  };

  return (
    <div
      onMouseEnter={handleHover}
      onMouseLeave={handleLeave}
      className={[PromptTagMain, type].join(' ')}>
      <p className={!showToggle ? PromptTagText : PromptTagToggle}>{name}</p>
      {showToggle && <button className={TagToggle} onClick={toggleType}>
        {type === 'positive' ? <i className={[IconFont, 'fa-solid', 'fa-minus'].join(" ")}></i> : <i className={[IconFont, 'fa-solid', 'fa-plus'].join(" ")}></i>}
      </button>}
      {showToggle && <button className={TagRemoveButton} onClick={handleRemove}>
        <i className={[IconFont, 'fa-solid', 'fa-close'].join(" ")}></i>
      </button>}
    </div>
  );
};