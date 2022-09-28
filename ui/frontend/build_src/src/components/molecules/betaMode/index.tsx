import React, { useState, useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import {
  KEY_CONFIG,
  getConfig,
  KEY_TOGGLE_CONFIG,
  toggleBetaConfig,
} from "../../../api";

import { useTranslation } from "react-i18next";

export default function BetaMode() {
  const { t } = useTranslation();
  // gate for the toggle
  const [shouldSetCofig, setShouldSetConfig] = useState(false);
  // next branch to get
  const [branchToGetNext, setBranchToGetNext] = useState("beta");

  // our basic config
  const { status: configStatus, data: configData } = useQuery(
    [KEY_CONFIG],
    getConfig
  );
  const queryClient = useQueryClient();

  // the toggle config
  const { status: toggleStatus, data: toggleData } = useQuery(
    [KEY_TOGGLE_CONFIG],
    async () => await toggleBetaConfig(branchToGetNext),
    {
      enabled: shouldSetCofig,
    }
  );

  // this is also in the Header Display
  // TODO: make this a custom hook
  useEffect(() => {
    if (configStatus === "success") {
      const { update_branch: updateBranch } = configData;

      if (updateBranch === "main") {
        setBranchToGetNext("beta");
      } else {
        // setIsBeta(true);
        setBranchToGetNext("main");
      }
    }
  }, [configStatus, configData]);

  useEffect(() => {
    if (toggleStatus === "success") {
      if (toggleData[0] === "OK") {
        // force a refetch of the config
        void queryClient.invalidateQueries([KEY_CONFIG])
      }
      setShouldSetConfig(false);
    }
  }, [toggleStatus, toggleData, setShouldSetConfig]);

  return (
    <label>
      <input
        type="checkbox"
        checked={branchToGetNext === "main"}
        onChange={(e) => {
          setShouldSetConfig(true);
        }}
      />🔥
      {t("advanced-settings.beta")} {t("advanced-settings.beta-disc")}
    </label>
  );
}
