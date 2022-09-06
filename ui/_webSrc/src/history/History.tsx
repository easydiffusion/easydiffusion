import React, {useState} from 'react'
import './History.css'
import {useLiveQuery} from "dexie-react-hooks";
import {historyDb, Item} from "./history-db";
import {HistoryFooter} from "./HistoryFooter";
import {HistoryItems} from "./HistoryItems";
import {Pagination} from "./Pagination";

export function History() {
	const [perPage, _setPerPageTODO] = useState(25);
	const [currentPage, setCurrentPage] = useState(1);
	const [showFavouritesOnly, setShowFavouritesOnly] = useState(false);
	const [filterQuery, setFilterQuery] = useState('');
	const offset = (perPage * currentPage) - perPage;

	const onClearFilters = Boolean(filterQuery || showFavouritesOnly) && function () {
		setFilterQuery('');
		setShowFavouritesOnly(false);
	}

	function collectionFilterPredicate(item: Item): boolean {
		if (showFavouritesOnly && !item.isFavourite) {
			return false;
		}
		if (!filterQuery) {
			return true;
		}

		return item.prompt.includes(filterQuery);
	}

	const historyItems = useLiveQuery(
		() => {
			return historyDb.items.orderBy('date').filter(collectionFilterPredicate).offset(offset).limit(perPage).toArray();
		},
		[offset, perPage, filterQuery, showFavouritesOnly],
		undefined
	);
	const totalItems = useLiveQuery(
		() => historyDb.items.filter(collectionFilterPredicate).count(),
		[filterQuery, showFavouritesOnly]
	)

	return (
		<section className="history-view history-view--temp-overlay">
			<div>
				<h2>History</h2>
				<div className="history-view__filters">
					<label>
						<input type="checkbox" checked={showFavouritesOnly} onChange={e => setShowFavouritesOnly(e.target.checked)} />
						Favourites only
					</label>
					<label>
						<input type="search" placeholder="Filter history" value={filterQuery} onChange={e => setFilterQuery(e.target.value)} />
						<span className="visually-hidden">Filter history</span>
					</label>
				</div>
			</div>
			<div className="history-view__body">
				<HistoryItems items={historyItems} clearFilters={onClearFilters}/>
				<Pagination perPage={perPage} currentPage={currentPage} totalItems={totalItems} onPageChange={newPage => setCurrentPage(newPage)} />
			</div>
			<HistoryFooter />
		</section>
	)
}
