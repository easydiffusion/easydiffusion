import React from "react";
import { useImageCreate } from "../../../stores/imageCreateStore";

interface ModifierTagProps {
  name: string;
}

export default function ModifierTag ({ name }: ModifierTagProps) {
  const hasTag = useImageCreate((state) => state.hasTag(name))
    ? "selected"
    : "";
  const toggleTag = useImageCreate((state) => state.toggleTag);

  const _toggleTag = () => {
    toggleTag(name);
  };

  return (
    <div className={"modifierTag " + hasTag} onClick={_toggleTag}>
      <p>{name}</p>
    </div>
  );
}
