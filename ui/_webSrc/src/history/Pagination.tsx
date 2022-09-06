import React from 'react'
import './Pagination.css';

export interface PaginationProps {
	perPage: number;
	currentPage: number;
	totalItems: number | undefined;
	onPageChange: (newPage: number) => void;
}

export function Pagination(props: PaginationProps) {
	const {
		perPage,
		currentPage,
		totalItems,
		onPageChange
	} = props;

	if (!totalItems) {
		return <></>;
	}

	const totalPages = Math.ceil(totalItems / perPage);
	if (totalPages <= 1) {
		return <></>;
	}

	const numPagesEitherSideToShow = 3;
	const minPage = Math.max(1, currentPage - numPagesEitherSideToShow);
	const maxPage = Math.min(totalPages, currentPage + numPagesEitherSideToShow);
	const pages = range(minPage, maxPage);

	function renderPage(page: number) {
		return (
			<button type="button" disabled={currentPage === page} onClick={() => onPageChange(page)}>{page}</button>
		)
	}

	return (
		<div className="pagination">
			<div>
				Page {currentPage} of {totalPages}
			</div>
			<ul>
				<li>
					<button type="button" disabled={currentPage === 1} onClick={() => onPageChange(currentPage - 1)}>Prev</button>
				</li>
				{minPage > 1 && (
					<>
						<li>
							{renderPage(1)}
						</li>
						<li>
							...
						</li>
					</>
				)}

				{pages.map(p => (
					<li key={p}>
						{renderPage(p)}
					</li>
				))}
				{maxPage < totalPages && (
					<>
						<li>
							...
						</li>
						<li>
							{renderPage(totalPages)}
						</li>
					</>
				)}
				<li>
					<button type="button" disabled={currentPage === totalPages} onClick={() => onPageChange(currentPage + 1)}>Next</button>
				</li>
			</ul>
			<div>&nbsp;</div>
		</div>
	)
}

function range(start: number, stop: number, step = 1) {
	return Array.from({length: (stop - start) / step + 1}, (_, i) => start + (i * step))
}