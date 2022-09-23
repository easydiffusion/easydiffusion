
import * as React from 'react';
import { useInfiniteQuery } from 'react-query';

interface CustomQueryParams {
  eventData: string[];
}

type Status = "success" | "loading" | "error";

export const useEventSourceQuery = (key: string, url: string, eventName: string) => {

  const eventSource = React.useRef<EventSource>(new EventSource(url));

  const queryFn = React.useCallback((_: any, params: CustomQueryParams) => {
    if (!params) { return Promise.resolve([]) }
    return Promise.resolve(params.eventData);
  }, [])

  const { data, fetchMore } = useInfiniteQuery<string[], string, CustomQueryParams>(key, queryFn as any, { getFetchMore: () => ({ eventData: [] }) })

  const customStatus = React.useRef<Status>('success');

  React.useEffect(() => {
    const evtSource = eventSource.current;

    const onEvent = function (ev: MessageEvent | Event) {
      if (!e.data) {
        return;
      }
      // Let's assume here we receive multiple data, ie. e.data is an array.
      fetchMore({ eventData: e.data });
    }
    const onError = () => { customStatus.current = 'error' };

    evtSource.addEventListener(eventName, onEvent);
    evtSource.addEventListener('error', onError);

    return () => {
      evtSource.removeEventListener(eventName, onEvent);
      evtSource.removeEventListener('error', onError);
    }

  }, [url, eventName, fetchMore])

  return { status: customStatus.current, data };
}