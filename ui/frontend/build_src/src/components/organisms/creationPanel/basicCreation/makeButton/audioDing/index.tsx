import React from "react";
import { API_URL } from "../../../../../../api";

const url = `${API_URL}/ding.mp3`;

const AudioDing = React.forwardRef((props, ref) => (
  // @ts-expect-error
  <audio ref={ref} style={{ display: "none" }}>
    <source src={url} type="audio/mp3" />
  </audio>
));

AudioDing.displayName = "AudioDing";

export default AudioDing;
