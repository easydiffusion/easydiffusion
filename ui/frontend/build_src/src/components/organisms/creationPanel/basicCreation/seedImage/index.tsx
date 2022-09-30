import React, { useRef, ChangeEvent } from "react";

import {
  ImageInputDisplay,
  InputLabel,
  ImageInput,
  ImageFixer,
  XButton,
} from "./seedImage.css";

import {
  buttonStyle
} from "../../../../_recipes/button.css";

import { useImageCreate } from "../../../../../stores/imageCreateStore";

import { useTranslation } from "react-i18next";

// TODO : figure out why this needs props to be passed in.. fixes a type error
// when the component is used in the parent component
export default function SeedImage(_props: any) {
  const { t } = useTranslation();

  const imageInputRef = useRef<HTMLInputElement>(null);

  const initImage = useImageCreate((state) =>
    state.getValueForRequestKey("init_image")
  );

  const isInPaintingMode = useImageCreate((state) => state.isInpainting);

  const setRequestOption = useImageCreate((state) => state.setRequestOptions);

  const _startFileSelect = () => {
    imageInputRef.current?.click();
  };
  const _handleFileSelect = (event: ChangeEvent<HTMLInputElement>) => {
    // @ts-expect-error
    const file = event.target.files[0];

    if (void 0 !== file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        if (e.target != null) {
          setRequestOption("init_image", e.target.result);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const toggleInpainting = useImageCreate((state) => state.toggleInpainting);

  const _handleClearImage = () => {
    setRequestOption("init_image", undefined);

    setRequestOption("mask", undefined);
    if (isInPaintingMode) {
      toggleInpainting();
    }
  };

  return (
    <div className={ImageInputDisplay}>
      <div>
        <label className={InputLabel}>
          <b>{t("home.initial-img-txt")}</b>
        </label>
        <input
          ref={imageInputRef}
          className={ImageInput}
          name="init_image"
          type="file"
          onChange={_handleFileSelect}
        />
        <button className={buttonStyle()} onClick={_startFileSelect}>
          {t("home.initial-img-btn")}
        </button>
      </div>

      <div className={ImageFixer}>
        {void 0 !== initImage && (
          <>
            <div>
              <img src={initImage} width="100" height="100" />
              <button className={XButton} onClick={_handleClearImage}>
                X
              </button>
            </div>
            <label>
              <input
                type="checkbox"
                onChange={(e) => {
                  toggleInpainting();
                }}
                checked={isInPaintingMode}
              ></input>
              {t("in-paint.txt")}
            </label>
          </>
        )}
      </div>
    </div>
  );
}