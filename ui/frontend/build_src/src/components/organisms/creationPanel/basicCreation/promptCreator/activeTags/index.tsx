import React from "react";

import { useImageCreate } from "../../../../../../stores/imageCreateStore";


import PromptTag from "../../../../../molecules/promptTag";

import {
  ActiveTagListMain
} from "./activeTags.css";


export default function ActiveTags() {

  const createTags = useImageCreate((state) => state.createTags);

  return (
    <div>
      <ul className={ActiveTagListMain}>
        {createTags.map((tag) => {
          console.log(tag);
          return (
            <li key={tag.id}>
              {/* @ts-expect-error */}
              <PromptTag id={tag.id} name={tag.name} category={tag?.category} previews={tag?.previews} type={tag.type} />
            </li>)
        }
        )}
      </ul>
    </div>
  );
}
