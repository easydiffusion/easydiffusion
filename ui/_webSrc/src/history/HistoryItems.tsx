import React from 'react'
import './HistoryItems.css'
import {Item} from "./history-db";
import {HistoryItem} from "./HistoryItem";

export interface HistoryItemsProps {
	items: Array<Item> | undefined;
	clearFilters: (() => void) | false;
}

export function HistoryItems(props: HistoryItemsProps) {
	const { items, clearFilters } = props;

	const showLoadingState = !items;
	const showEmptyState = items && !items.length;
	const showContent = !showLoadingState && !showEmptyState;

	return (
		<div className="history-items">
			{showLoadingState && (
				<div>Loading your history...</div>
			)}
			{showEmptyState && (
				<div>
					<p>You don't seem to have any history yet!</p>
					<p>Results will show up here when you tick "<strong>Automatically save to disk</strong>" when making images.</p>
					{clearFilters && (
						<p>You could try to <button type="button" onClick={clearFilters}>clear current filters</button>.</p>
					)}
				</div>
			)}
			{showContent && (
				<div>
					<ul className="history-items__grid">
						{items?.map(item => <li key={item.id}>
							<HistoryItem item={item} />
						</li>)}
					</ul>
				</div>
			)}
		</div>
	)
}
