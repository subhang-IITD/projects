import heapq

def dijkstra_max_capacity(graph, start, end):
    pq = [(-float('inf'), start)]
    max_capacity = {start: float('inf')}

    while pq:
        capacity, node = heapq.heappop(pq)
        capacity = -capacity

        if node == end:
            return capacity

        for neighbor, weight in graph[node]:
            min_capacity = min(capacity, weight)
            if neighbor not in max_capacity or min_capacity > max_capacity[neighbor]:
                max_capacity[neighbor] = min_capacity
                heapq.heappush(pq, (-min_capacity, neighbor))

    return 0

graph = {
    'A': [('B', 5), ('C', 10)],
    'B': [('D', 7)],
    'C': [('D', 15)],
    'D': [('E', 8)],
    'E': []
}

start_node = 'A'
end_node = 'E'
print("Maximum Capacity from", start_node, "to", end_node, "is", dijkstra_max_capacity(graph, start_node, end_node))
