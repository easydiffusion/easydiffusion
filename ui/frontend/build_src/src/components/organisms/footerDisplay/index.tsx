import React from "react";

import "./footerDisplay.css";

import { API_URL } from "../../../api";

export default function FooterDisplay() {
  return (
    <div id="footer" className="panel-box">
      <p>
        If you found this project useful and want to help keep it alive, please{" "}
        <a href="https://ko-fi.com/cmdr2_stablediffusion_ui" target="_blank">
          <img src={`${API_URL}/kofi.png`} id="coffeeButton" />
        </a>{" "}
        to help cover the cost of development and maintenance! Thank you for
        your support!
      </p>
      <p>
        Please feel free to join the{" "}
        <a href="https://discord.com/invite/u9yhsFmEkB" target="_blank">
          discord community
        </a>{" "}
        or{" "}
        <a
          href="https://github.com/cmdr2/stable-diffusion-ui/issues"
          target="_blank"
        >
          file an issue
        </a>{" "}
        if you have any problems or suggestions in using this interface.
      </p>
      <div id="footer-legal">
        <p>
          <b>Disclaimer:</b> The authors of this project are not responsible for
          any content generated using this interface.
        </p>
        <p>
          This license of this software forbids you from sharing any content
          that violates any laws, produce any harm to a person, disseminate any
          personal information that would be meant for harm, <br />
          spread misinformation and target vulnerable groups. For the full list
          of restrictions please read{" "}
          <a
            href="https://github.com/cmdr2/stable-diffusion-ui/blob/main/LICENSE"
            target="_blank"
          >
            the license
          </a>
          .
        </p>
        <p>
          By using this software, you consent to the terms and conditions of the
          license.
        </p>
      </div>
    </div>
  );
}
