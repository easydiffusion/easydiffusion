import { useQuery, useQueryClient } from 'react-query'

export const useEventSourceQuery = (queryKey, url, eventName) => {
  const queryClient = useQueryClient()

  const fetchData = () => {
    const evtSource = new EventSource(url)

    evtSource.addEventListener(eventName, (event) => {
      const eventData = event.data && JSON.parse(event.data)

      if (eventData) {
        queryClient.setQueryData(queryKey, eventData)
      }
    })
  }

  return useQuery(queryKey, fetchData)
}