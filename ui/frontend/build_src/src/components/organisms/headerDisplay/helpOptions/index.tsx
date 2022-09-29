import React from "react";
import { Popover } from '@headlessui/react';
// import { useTranslation } from "react-i18next";

import {
  PopoverMain,
  PopoverButtonStyle,
  PopoverPanelMain,
} from "../../../_headless/popover/index.css";

import {
  IconFont,
  SettingItem
} from "../../../../styles/shared.css";

import {
  HelpContent
} from "./helpOptions.css";

export default function HelpOptions() {

  return (
    <Popover className={PopoverMain}>
      <Popover.Button className={PopoverButtonStyle}>
        <i className={[IconFont, 'fa-solid', 'fa-comments'].join(" ")}></i>
        Help & Community
      </Popover.Button>

      <Popover.Panel className={PopoverPanelMain}>
        <div className={HelpContent}>
          <ul>
            <li className={SettingItem}>
              <a href="https://github.com/cmdr2/stable-diffusion-ui/blob/main/Troubleshooting.md" target="_blank" rel="noreferrer">
                <i className={[IconFont, 'fa-solid', 'fa-circle-question'].join(" ")}></i> Usual Problems and Solutions
              </a>
            </li>
            <li className={SettingItem}>
              <a href="https://discord.com/invite/u9yhsFmEkB" target="_blank" rel="noreferrer">
                <i className={[IconFont, 'fa-brands', 'fa-discord'].join(" ")}></i> Discord user Community
              </a>
            </li>
            <li className={SettingItem}>
              <a href="https://old.reddit.com/r/StableDiffusionUI/" target="_blank" rel="noreferrer">
                <i className={[IconFont, 'fa-brands', 'fa-reddit'].join(" ")}></i> Reddit Community
              </a>
            </li>
            <li className={SettingItem}>
              <a href="https://github.com/cmdr2/stable-diffusion-ui " target="_blank" rel="noreferrer">
                <i className={[IconFont, 'fa-brands', 'fa-github'].join(" ")}></i> Source Code on Github
              </a>
            </li>
          </ul>
        </div>
      </Popover.Panel>
    </Popover>
  );
};