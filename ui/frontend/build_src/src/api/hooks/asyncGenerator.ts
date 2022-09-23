import { useInfiniteQuery } from "@tanstack/react-query";

export const useAyncGeneratorQuery = (key: string, asyncGeneratorFn: () => Generator<string[], string[], unknown>) => {

  const queryFn = React.useCallback((_: any, params: CustomQueryParams) => {
    if (!params) { return Promise.resolve([]) }
    return Promise.resolve(params.eventData);
  }, [])

  const { data, fetchMore } = useInfiniteQuery<string[], string, CustomQueryParams>(key, queryFn as any, { getFetchMore: () => ({ eventData: [] }) })

  const customStatus = React.useRef<Status>('success');

  React.useEffect(() => {
    (async function doReceive() {
      try {
        for await (let data of asyncGeneratorFn()) {
          fetchMore({ eventData: data });
        }
      } catch (e) {
        customStatus.current = 'error';
      }
    })();

  }, [asyncGeneratorFn, fetchMore])

  return { status: customStatus.current, data };
}