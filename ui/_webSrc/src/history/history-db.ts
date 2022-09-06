import Dexie, {Table} from 'dexie';

export interface ResponseItem {
	data?: string; // base64 image data
	url: string;
	seed: number;
}

export interface Response {
	status: 'succeeded' | 'error';
	output: Array<ResponseItem>;
}

export interface ItemRequest {
	prompt: string;
	num_outputs: number;
	num_inference_steps: number;
	guidance_scale: number;
	width: number;
	height: number;
	save_to_disk: boolean;
	turbo: boolean;
	use_cpu: boolean;
	use_full_precision: boolean;
	seed: number;
}


export interface Item extends ItemRequest {
	id: string;
	date: Date;
	isFavourite: 0 | 1; // cannot index boolean with DexieJS so using a number
	response?: Response;
}

export class HistoryDb extends Dexie {
	items!: Table<Item>;

	constructor() {
		super('stable-diffusion-ui-history');

		this.version(2).stores({
			items: '++id, prompt, date, isFavourite' // Primary key and indexed props
		});
	}
}

export const historyDb = new HistoryDb();

// TODO when rest of UI is ported to react, replace global hooks with actual react code.
const w = (window as any);
w.addItem = async (item: Item) => {
	if (!item.save_to_disk) {
		return;
	}
	item.id = crypto.randomUUID();
	item.date = new Date();

	await historyDb.items.add(item);

	return item.id;
}
w.updateItem = async (itemId: string, response: Response) => {
	const item = await historyDb.items.get(itemId);
	if (!item?.save_to_disk) {
		return;
	}

	const {output} = response;
	const outputWithoutBase64ImageData = output.map(responseItem => {
		const {data, ...responseWithoutBase64ImageData} = responseItem;
		return responseWithoutBase64ImageData;
	});

	// Don't store base64 image data in IndexDB due to size - instead only save URL to file on disk.
	await historyDb.items.update(item.id, {
		response: {
			...response,
			output: outputWithoutBase64ImageData
		}
	});
}