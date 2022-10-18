import React, { useState, KeyboardEventHandler, Fragment } from "react";
import { Switch } from '@headlessui/react'

import { useImageCreate } from "../../../../../stores/imageCreateStore";
import { usePromptMatrix } from "../../../../../stores/promptMatrixStore";
import { v4 as uuidv4 } from "uuid";
import PromptTag from "../../../../molecules/promptTag";

import ActiveTags from "../promptCreator/activeTags";

import MatrixList from "./matrixList";
import {
  IconFont,
} from "../../../../../styles/shared.css";

import {
  ToggleGroupMain,
  ToggleMain,
  TogglePill,
  inputRow,
  prmptBtn,
} from "../promptCreator/promptCreator.css";

import {
  buttonStyle,
} from "../../../../_recipes/button.css";
// import Checkbox from "src/components/atoms/headlessCheckbox";

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

export default function ModificationPanel() {

  const [positive, setPositive] = useState(true)

  const potentialTags = usePromptMatrix((state) => state.potentialTags);
  const setPotentialTags = usePromptMatrix((state) => state.setPotentialTags);
  const generateTags = usePromptMatrix((state) => state.generateTags);

  const enterPrompt = () => {
    if (potentialTags !== '') {
      generateTags(positive);
    }
  }

  const checkForEnter = (event: KeyboardEventHandler<HTMLInputElement>) => {
    // @ts-expect-error
    if (event.key === 'Enter') {
      enterPrompt();
    }
  };

  return (
    <div>
      <div className=''>
        <span>
          <label> modifiers</label>
          {/* @ts-expect-error */}
          <input value={potentialTags} onKeyDown={checkForEnter} onChange={(event) => {
            setPotentialTags(event.target.value);
          }}></input>
        </span>
        <TagTypeToggle positive={positive} setPositive={setPositive}></TagTypeToggle>

        <button
          className={[buttonStyle({
            color: 'secondary',
            size: 'large',
          })].join(" ")}
          onClick={enterPrompt}
        >
          Add Modification
        </button>
      </div>

      <MatrixList></MatrixList>
      {/*
      <ActiveTags></ActiveTags> */}

    </div>
  );
}