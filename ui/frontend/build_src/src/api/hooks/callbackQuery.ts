type CallbackType = (data: string[]) => void;

type InitCallbackType = (cb: CallbackType) => void;

export const useCallbackQuery = (key: string, initCallbackQuery: InitCallbackType) => {

  const queryFn = React.useCallback((_: any, params: CustomQueryParams) => {
    if (!params) { return Promise.resolve([]) }
    return Promise.resolve(params.eventData);
  }, [])

  const { data, fetchMore } = useInfiniteQuery<string[], string, CustomQueryParams>(key, queryFn as any, { getFetchMore: () => ({ eventData: [] }) })

  const callback = React.useCallback((data) => { fetchMore({ eventData: data }) }, [fetchMore]);

  React.useEffect(() => { initCallbackQuery(callback); }, [callback, initCallbackQuery])

  const customStatus = React.useRef<Status>('success');

  return { status: customStatus.current, data };
}