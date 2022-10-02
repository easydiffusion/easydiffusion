import React, { useState, ChangeEvent, KeyboardEventHandler, Fragment } from "react";
import { v4 as uuidv4 } from "uuid";
import { Switch } from '@headlessui/react'

import { useImageCreate } from "../../../../../stores/imageCreateStore";

import ActiveTags from "./activeTags";

import {
  IconFont,
} from "../../../../../styles/shared.css";

import {
  buttonStyle,
} from "../../../../_recipes/button.css";

import {
  PromptCreatorMain,
  ToggleGroupMain,
  ToggleMain,
  ToggleLabel,
  ToggleEnabled,
  TogglePill,
  buttonRow,
} from "./promptCreator.css";



import { useTranslation } from "react-i18next";
import { type } from "os";

interface TagTypeProps {
  positive: boolean;
  setPositive: (positive: boolean) => void;
};

function TagTypeToggle({ positive, setPositive }: TagTypeProps) {
  return (
    <Switch.Group as={Fragment}>
      <div className={ToggleGroupMain}>
        <Switch.Label> Type </Switch.Label>
        <Switch className={ToggleMain} checked={positive} onChange={setPositive}>
          <span
            className={TogglePill}
          >
            {positive
              ? <i className={[IconFont, 'fa-solid', 'fa-plus'].join(" ")}></i>
              : <i className={[IconFont, 'fa-solid', 'fa-minus'].join(" ")}></i>}
          </span>
        </Switch>
      </div>
    </Switch.Group>
  );
}

export default function PromptCreator() {

  const [positive, setPositive] = useState(true)
  const [tagText, setTagText] = useState('An astronaut riding a horse');

  const addCreateTag = useImageCreate((state) => state.addCreateTag);

  const { t } = useTranslation();

  const checkForEnter = (event: KeyboardEventHandler<HTMLInputElement>) => {
    // @ts-expect-error
    if (event.key === "Enter") {
      if (tagText !== '') {
        const type = positive ? "positive" : "negative";

        tagText.split(',').map((tag) => tag.trim()).forEach((tag) => {
          addCreateTag({ id: uuidv4(), name: tag, type });
        });
        //debugger;

        setTagText('');
      }
    }
  };

  return (
    <div className={PromptCreatorMain}>
      <div>
        <p>{t("home.editor-title")}</p>
        {/* @ts-expect-error */}
        <input value={tagText} onKeyDown={checkForEnter} onChange={(event) => {
          setTagText(event.target.value)
        }}></input>
      </div>
      <div className={buttonRow}>
        <button
          className={buttonStyle(
            {
              size: 'slim'
            }
          )}
          onClick={() => {
          }}
        >
          Add Prompt
        </button>

        <TagTypeToggle positive={positive} setPositive={setPositive}></TagTypeToggle>
      </div>
      <ActiveTags></ActiveTags>
    </div >
  );
}