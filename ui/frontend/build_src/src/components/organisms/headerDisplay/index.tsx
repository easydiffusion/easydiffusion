import React, { useEffect, useState } from "react";

import { useQuery } from "@tanstack/react-query";
import { KEY_CONFIG, getConfig } from "@api/index";

import StatusDisplay from "./statusDisplay";

import HelpOptions from "./helpOptions";
import SystemSettings from "./systemSettings";

import { useTranslation } from "react-i18next";

import {
  HeaderDisplayMain,
  HeaderTitle,
  HeaderLinks,
} from "./headerDisplay.css";

// import LanguageDropdown from "./languageDropdown";

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
      const { update_branch: updateBranch } = data;

      // just hard coded for now
      setVersion("v2.1");

      if (updateBranch === "main") {
        setRelease("(stable)");
      } else {
        setRelease("(beta)");
      }
    }
  }, [status, data, setVersion, setVersion]);

  return (
    <div className={HeaderDisplayMain}>
      <div className={HeaderTitle}>
        <h1>
          {t("title")} {version} {release}{" "}
        </h1>
        <StatusDisplay className="status-display"></StatusDisplay>
      </div>
      <div className={HeaderLinks}>
        <HelpOptions></HelpOptions>
        <SystemSettings></SystemSettings>
      </div>
      {/* <LanguageDropdown></LanguageDropdown> */}
    </div>
  );
}
