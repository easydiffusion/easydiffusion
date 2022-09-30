import React from "react";
import { Tab } from '@headlessui/react';

import CreationPanel from "../../organisms/creationPanel";
import QueueDisplay from "../../organisms/queueDisplay";

import {
  CreationTabsMain
} from "./creationTabs.css";

export default function CreationTabs() {

  return (
    <Tab.Group>
      <Tab.List>
        <Tab>Create</Tab>
        <Tab>Queue</Tab>
      </Tab.List>
      <Tab.Panels className={CreationTabsMain}>
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
