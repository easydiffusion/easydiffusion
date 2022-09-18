import React, { useState, useEffect } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';

import {
  KEY_CONFIG, getConfig,
  KEY_TOGGLE_CONFIG, toggleBetaConfig
} from '../../../api';





export default function BetaMode() {

  // gate for the toggle
  const [shouldSetCofig, setShouldSetConfig] = useState(false);
  // next branch to get 
  const [branchToGetNext, setBranchToGetNext] = useState("beta");

  // our basic config
  const { status: configStatus, data: configData } = useQuery([KEY_CONFIG], getConfig);
  const queryClient = useQueryClient()

  // the toggle config
  const { status: toggleStatus, data: toggleData } = useQuery([KEY_TOGGLE_CONFIG], () => toggleBetaConfig(branchToGetNext), {
    enabled: shouldSetCofig,
  });


  // this is also in the Header Display
  // TODO: make this a custom hook
  useEffect(() => {

    if (configStatus === 'success') {

      const { update_branch } = configData;

      if (update_branch === 'main') {
        setBranchToGetNext('beta');
      } else {
        // setIsBeta(true);
        setBranchToGetNext('main');
      }

    }
  }, [configStatus, configData]);


  useEffect(() => {
    if (toggleStatus === 'success') {

      if (toggleData[0] == 'OK') {
        // force a refetch of the config
        queryClient.invalidateQueries([KEY_CONFIG])
      }
      setShouldSetConfig(false);

    }
  }, [toggleStatus, toggleData, setShouldSetConfig]);



  return (
    <label>
      <input
        type="checkbox"
        checked={branchToGetNext === 'main'}
        onChange={(e) => {
          setShouldSetConfig(true);
        }}
      />
      Enable Beta Mode
    </label>
  );
}

