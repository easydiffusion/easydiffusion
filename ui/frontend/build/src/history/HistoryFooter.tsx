import React from "react";
import "./HistoryFooter.css";
import { historyDb } from "./history-db";

export function HistoryFooter() {
  async function onClearAllHistory() {
    const confirmMessage = `Are you sure you want to clear all Stable Diffusion UI history data from this browser?

All prompts and results will be removed.

NOTE: This does NOT remove image data from your local folder, you will need to remove this separately.`;

    if (window.confirm(confirmMessage)) {
      await historyDb.items.clear();
    }
  }

  return (
    <div className="history-view__footer">
      Your history is stored locally in this browser. Images are stored on disk
      at your configured location.
      <br />
      <button
        type="button"
        className="history-view__clear-history-btn"
        onClick={onClearAllHistory}
      >
        Clear all history
      </button>
    </div>
  );
}
