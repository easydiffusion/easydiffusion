import React, { useState } from "react";


import {
  PanelBox,
  MenuButton,
} from "../../../../styles/shared.css";

import {
  card
} from '../../../_recipes/card.css';

import {
  buttonStyle,
} from "../../../_recipes/button.css";

import {
  ImagerModifierGroups,
  ImageModifierGrouping,
  ModifierListStyle,
} from "./imageModifiers.css";

import { ModifierObject, useImageCreate } from "../../../../stores/imageCreateStore";
import { useCreateUI } from "../creationPanelUIStore";

import ModifierTag from "../../../molecules/modifierTag";

interface ModifierListProps {
  category: string;
  tags: ModifierObject[];
}

function ModifierList({ tags, category }: ModifierListProps) {
  return (
    <ul className={ModifierListStyle}>
      {tags.map((tag) => (
        <li key={tag.modifier}>
          <ModifierTag category={category} name={tag.modifier} previews={tag.previews} />
        </li>
      ))}
    </ul>
  );
}

interface ModifierGroupingProps {
  title: string;
  category: string;
  tags: ModifierObject[];
}

function ModifierGrouping({ title, category, tags }: ModifierGroupingProps) {
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
      {isExpanded && <ModifierList category={category} tags={tags} />}
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
    <div className={card(
      {
        level: 1,
        baking: 'normal'
      }
    )}>
      <button
        type="button"
        onClick={handleClick}
        className={buttonStyle({
          type: 'clear',
          size: 'large'
        })}
      >
        Image Modifiers
      </button>

      {imageModifierIsOpen && (
        <ul className={ImagerModifierGroups}>
          {allModifiers.map((item, index) => {
            return (
              <li key={item.category}>
                <ModifierGrouping title={item.category} category={item.category} tags={item.modifiers} />
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
