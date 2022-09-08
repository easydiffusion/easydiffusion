import React, { useState } from "react";
import "./HistoryItem.css";
import { historyDb, Item } from "./history-db";
import { CloseIcon, DownloadIcon } from "../icons";
import { TrashIcon } from "../icons/TrashIcon";
import { StarIcon } from "../icons/StarIcon";

export interface HistoryItemProps {
  item: Item;
}

export function HistoryItem(props: HistoryItemProps) {
  const { item } = props;
  const { response } = item;
  const showErrorState =
    !response || response.status !== "succeeded" || !response.output.length;
  if (showErrorState) {
    console.warn(
      "Unable to render a history item, maybe it failed to generate or has not yet finished? Item: ",
      item
    );
  }
  const output = response?.output[0];

  const [itemDetailsAreVisible, setItemDetailsAreVisible] = useState(false);

  async function toggleFavouriteState(newValue: boolean) {
    if (!item?.id) {
      return;
    }

    await historyDb.items.update(item.id, { isFavourite: newValue ? 1 : 0 });
  }

  async function deleteItem() {
    if (!item?.id) {
      return;
    }

    const confirmMessage = `Are you sure you want to delete the item?

You will not be able to recover this data.`;

    if (window.confirm(confirmMessage)) {
      await historyDb.items.delete(item.id);
    }
  }

  function onDownloadImage() {
    alert("Not implemented yet! Right click -> Save As for now...");
  }

  return (
    <div
      className="history-item"
      style={{ maxWidth: `${item.width}px`, maxHeight: `${item.height}px` }}
    >
      {showErrorState && (
        <div>
          Unable to show this item. Maybe it failed to generate, or has not yet
          finished?
          <br />
          <small>
            Check browser devtools console for details of this item.
          </small>
        </div>
      )}
      {output && (
        <>
          <button
            className="history-item__toggle-details-btn"
            type="button"
            onClick={() => setItemDetailsAreVisible((state) => !state)}
          >
            <span className="visually-hidden">Show item details</span>
          </button>
          <div className="history-item__body-wrapper">
            <img src={`/image/${output.url}`} alt={item.prompt} />
            {itemDetailsAreVisible && (
              <div className="history-item__details">
                <div className="history-item__details-actions">
                  <button
                    type="button"
                    className={
                      "history-item__details-actions-favourite" +
                      (item.isFavourite
                        ? " history-item__details-actions-favourite--active"
                        : "")
                    }
                    title={
                      item.isFavourite
                        ? "Remove from favourites"
                        : "Add to favourites"
                    }
                    onClick={() => toggleFavouriteState(!item.isFavourite)}
                  >
                    <StarIcon
                      filled={Boolean(item.isFavourite)}
                      label={
                        item.isFavourite
                          ? "Remove from favourites"
                          : "Add to favourites"
                      }
                    />
                  </button>
                  <button
                    type="button"
                    onClick={() => setItemDetailsAreVisible(false)}
                    title="Hide item details"
                  >
                    <CloseIcon label="Hide item details" />
                  </button>
                </div>
                <dl className="history-item__details-body">
                  <dt>generated</dt>
                  <dd>{item.date.toLocaleString()}</dd>
                  <dt>seed</dt>
                  <dd>{item.seed}</dd>
                  <dt>dimensions</dt>
                  <dd>
                    {item.width}px * {item.height}px
                  </dd>
                  <dt>guidance_scale</dt>
                  <dd>{item.guidance_scale}</dd>
                  <dt>num_inference_steps</dt>
                  <dd>{item.num_inference_steps}</dd>
                </dl>
                <div className="history-item__details-actions">
                  <button
                    type="button"
                    onClick={() => onDownloadImage()}
                    title="Download image"
                  >
                    <DownloadIcon label="Download image" />
                  </button>

                  <button
                    type="button"
                    className="history-item__details-actions-delete"
                    onClick={deleteItem}
                    title="Delete this item"
                  >
                    <TrashIcon label="Delete this item" />
                  </button>
                </div>
              </div>
            )}
          </div>
          <div className="history-item__prompt">{item.prompt}</div>
        </>
      )}
    </div>
  );
}
