import React from "react";

import { useRef } from "react";
import { API_URL } from "../../../api";

const url = `${API_URL}/ding.mp3`;

const AudioDing = React.forwardRef((props, ref) => (
  <audio ref={ref} style={{display:'none'}}>
    <source src={url} type="audio/mp3"/>
  </audio>
));

export default AudioDing;
  
//   {
//   const url = `${API_URL}/ding.mp3`;
//   const audioRef = useRef<HTMLAudioElement>(null);

//   return (
//       <audio ref={audioRef} style={{display:'none'}}>
//         <source src={url} type="audio/mp3"/>
//       </audio>
//   );
// }
