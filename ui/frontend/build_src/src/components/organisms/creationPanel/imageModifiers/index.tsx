import React, { useState } from "react";

// @ts-expect-error
import { PanelBox } from "../../../../styles/shared.css.ts";

import {
  ImagerModifierGroups,
  ImageModifierGrouping,
  MenuButton,
  ModifierListStyle, //@ts-expect-error
} from "./imageModifiers.css.ts";

import { useImageCreate } from "../../../../stores/imageCreateStore";
import { useCreateUI } from "../creationPanelUIStore";

import ModifierTag from "../../../atoms/modifierTag";

interface ModifierListProps {
  tags: string[];
}

function ModifierList({ tags }: ModifierListProps) {
  return (
    <ul className={ModifierListStyle}>
      {tags.map((tag) => (
        <li key={tag}>
          <ModifierTag name={tag} />
        </li>
      ))}
    </ul>
  );
}

interface ModifierGroupingProps {
  title: string;
  tags: string[];
}

function ModifierGrouping({ title, tags }: ModifierGroupingProps) {
  // doing this localy for now, but could move to a store
  // and persist if we wanted to
  const [isExpanded, setIsExpanded] = useState(false);

  const _toggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div className={ImageModifierGrouping}>
      <button type="button" className={MenuButton} onClick={_toggleExpand}>
        <h4>{title}</h4>
      </button>
      {isExpanded && <ModifierList tags={tags} />}
    </div>
  );
}

export default function ImageModifers() {
  const allModifiers = useImageCreate((state) => state.allModifiers);

  const imageModifierIsOpen = useCreateUI((state) => state.isOpenImageModifier);
  const toggleImageModifiersIsOpen = useCreateUI(
    (state) => state.toggleImageModifier
  );

  const handleClick = () => {
    toggleImageModifiersIsOpen();
  };

  return (
    <div className={PanelBox}>
      <button
        type="button"
        onClick={handleClick}
        className="panel-box-toggle-btn"
      >
        {/* TODO: swap this manual collapse stuff out for some UI component? */}
        <h3>Image Modifiers (art styles, tags, ect)</h3>
      </button>

      {imageModifierIsOpen && (
        <ul className={ImagerModifierGroups}>
          {allModifiers.map((item, index) => {
            return (
              // @ts-expect-error
              <li key={item[0]}>
                {/* @ts-expect-error */}
                <ModifierGrouping title={item[0]} tags={item[1]} />
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
