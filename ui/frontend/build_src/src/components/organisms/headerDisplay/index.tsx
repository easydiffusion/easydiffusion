import React, { useEffect, useState } from "react";

import { useQuery } from "@tanstack/react-query";
import { KEY_CONFIG, getConfig } from "../../../api";

import StatusDisplay from "./statusDisplay";

import { useTranslation } from "react-i18next";

import {
  HeaderDisplayMain, // @ts-expect-error
} from "./headerDisplay.css.ts";

export default function HeaderDisplay() {
  const { t } = useTranslation();

  const { status, data } = useQuery([KEY_CONFIG], getConfig);

  const [version, setVersion] = useState("2.1.0");
  const [release, setRelease] = useState("");

  // this is also in the Beta Mode
  // TODO: make this a custom hook
  useEffect(() => {
    if (status === "success") {
      // TODO also pass down the actual version
      const { update_branch } = data;

      // just hard coded for now
      setVersion("v2.1");

      if (update_branch === "main") {
        setRelease("(stable)");
      } else {
        setRelease("(beta)");
      }
    }
  }, [status, data, setVersion, setVersion]);

  return (
    <div className={HeaderDisplayMain}>
      <h1>
        {t("title")} {version} {release}{" "}
      </h1>
      <StatusDisplay className="status-display"></StatusDisplay>
    </div>
  );
}
