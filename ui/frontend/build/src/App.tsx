import React, { useState } from "react";
import "./App.css";
import { History } from "./history";

function App() {
  const [historyIsVisible, setHistoryIsVisible] = useState(false);

  return (
    <div className="App">
      <button
        className="history-toggle-btn"
        type="button"
        onClick={() => setHistoryIsVisible((state) => !state)}
      >
        {historyIsVisible ? "Hide" : "Show"} history
      </button>
      {historyIsVisible && <History />}
    </div>
  );
}

export default App;
