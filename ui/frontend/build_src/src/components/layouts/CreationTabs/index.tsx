import React, { Fragment } from "react";
import { Tab } from '@headlessui/react';
import CreationPanel from "../../organisms/creationPanel";
import QueueDisplay from "../../organisms/queueDisplay";

import {
  CreationTabsMain
} from "./creationTabs.css";

import {
  tabPanelStyles,
  tabStyles,
} from "../../_recipes/tabs_headless.css";

import {
  card as cardStyle
} from "../../_recipes/card.css";



export default function CreationTabs() {

  return (
    <Tab.Group>
      <Tab.List>
        <Tab as={Fragment}>
          {({ selected }) => (
            <button
              className={tabStyles({
                selected,
              })}
            >
              Create
            </button>
          )}
        </Tab>

        <Tab as={Fragment}>
          {({ selected }) => (

            <button
              className={tabStyles({
                selected,
              })}
            >
              Queue
            </button>
          )}
        </Tab>


      </Tab.List>
      <Tab.Panels className={tabPanelStyles()}>
        <Tab.Panel>
          <CreationPanel></CreationPanel>
        </Tab.Panel>
        <Tab.Panel>
          <QueueDisplay></QueueDisplay>
        </Tab.Panel>
      </Tab.Panels>
    </Tab.Group>
  );
}
