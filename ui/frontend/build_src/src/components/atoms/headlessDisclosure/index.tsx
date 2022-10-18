import React from "react";
import { Disclosure } from '@headlessui/react'
import {
  buttonStyle,
} from "../../_recipes/button.css";

interface Props {
  buttonText: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  setOpenPersist?: (isOpen: boolean) => void;
}

export default function HeadlessDisclosure({ buttonText, children, defaultOpen, setOpenPersist }: Props) {

  return (
    <Disclosure defaultOpen={defaultOpen}>
      {({ open }) => {

        if (setOpenPersist != null) {
          setOpenPersist(open);
        }

        return (
          <>
            <Disclosure.Button className={buttonStyle({
              type: 'action',
              color: 'accent',
            })}
            >
              {buttonText}
            </Disclosure.Button>
            <Disclosure.Panel>
              {children}
            </Disclosure.Panel>
          </>
        )
      }}
    </Disclosure>
  );
}