import React from "react";
import { Tab } from '@headlessui/react';

import CreationPanel from "../../organisms/creationPanel";
import QueueDisplay from "../../organisms/queueDisplay";

import {
  CreationTabsMain
} from "./creationTabs.css";

import {
  card as cardStyle
} from "../../_recipes/card.css";

export default function CreationTabs() {

  return (
    <Tab.Group>
      <Tab.List>
        <Tab>Create</Tab>
        <Tab>Queue</Tab>
      </Tab.List>
      <Tab.Panels className={cardStyle({
        baking: 'normal',
      })}>
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
