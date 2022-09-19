import React, { useEffect, useState } from "react";

import { useQuery } from "@tanstack/react-query";
import { loadModifications } from "../../../../api";

import { useImageCreate } from "../../../../stores/imageCreateStore";
import { useCreateUI } from "../creationPanelUIStore";

import ModifierTag from "../../../atoms/modifierTag";

interface ModifierListProps {
  tags: string[];
}

function ModifierList({ tags }: ModifierListProps) {
  return (
    <ul className="modifier-list">
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
    <div className="modifier-grouping">
      <div className="modifier-grouping-header" onClick={_toggleExpand}>
        <h5>{title}</h5>
      </div>
      {isExpanded && <ModifierList tags={tags} />}
    </div>
  );
}

export default function ImageModifers() {
  // const { status, data } = useQuery(["modifications"], loadModifications);

  // const imageModifierIsOpen = useImageCreate(
  //   (state) => state.uiOptions.imageModifierIsOpen
  // );
  // const toggleImageModifiersIsOpen = useImageCreate(
  //   (state) => state.toggleImageModifiersIsOpen
  // );

  const allModifiers = useImageCreate((state) => state.allModifiers);

  console.log("allModifiers", allModifiers);

  const imageModifierIsOpen = useCreateUI((state) => state.isOpenImageModifier);
  const toggleImageModifiersIsOpen = useCreateUI(
    (state) => state.toggleImageModifier
  );

  const handleClick = () => {
    toggleImageModifiersIsOpen();
  };

  return (
    <div className="panel-box">
      <button
        type="button"
        onClick={handleClick}
        className="panel-box-toggle-btn"
      >
        {/* TODO: swap this manual collapse stuff out for some UI component? */}
        <h4>Image Modifiers (art styles, tags, ect)</h4>
      </button>

      {/* @ts-ignore */}
      {imageModifierIsOpen &&
        // @ts-ignore
        allModifiers.map((item, index) => {
          return (
            // @ts-ignore
            <ModifierGrouping key={item[0]} title={item[0]} tags={item[1]} />
          );
        })}
    </div>
  );
}
