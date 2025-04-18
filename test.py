import heapq

def intercept(roads, stations, start, friendStart):
    """
    Function to determine the optimal path to intercept a friend on a train loop.
    
    Args:
        roads: List of tuples (start_location, end_location, cost, time)
        stations: List of tuples (station_location, time_to_next_station)
        start: Your starting location
        friendStart: Your friend's starting train station
    
    Returns:
        Tuple (totalCost, totalTime, route) or None if interception is impossible
        
    Time Complexity: O(|R| log |L|)
    Space Complexity: O(|L| + |R|)
    """
    # Find the maximum location ID directly to determine the size of our arrays
    max_location = -1
    for road in roads:
        max_location = max(max_location, road[0], road[1])
    
    # Create an adjacency list for the road network using lists
    graph = [[] for _ in range(max_location + 1)]
    for start_loc, end_loc, cost, time in roads:
        graph[start_loc].append((end_loc, cost, time))
    
    # Extract station locations as a list
    station_locations = [station[0] for station in stations]
    
    # Find the index of friendStart in stations
    friend_start_idx = -1
    for i, station in enumerate(stations):
        if station[0] == friendStart:
            friend_start_idx = i
            break
    
    # Calculate the total time for a complete loop
    loop_time = sum(station[1] for station in stations)
    
    # Pre-compute friend's position for each time point in one complete loop
    friend_positions_loop = [None] * (loop_time + 1)
    friend_at_station_loop = [False] * (loop_time + 1)
    
    curr_time = 0
    curr_idx = friend_start_idx
    friend_positions_loop[curr_time] = station_locations[curr_idx]
    friend_at_station_loop[curr_time] = True
    
    # Pre-compute friend's positions for one complete loop
    while curr_time < loop_time:
        travel_time = stations[curr_idx][1]
        next_idx = (curr_idx + 1) % len(stations)
        
        # Friend is in transit between stations
        for t in range(1, travel_time):
            curr_time += 1
            if curr_time <= loop_time:
                friend_positions_loop[curr_time] = None
                friend_at_station_loop[curr_time] = False
        
        # Friend arrives at next station
        curr_time += 1
        if curr_time <= loop_time:
            friend_positions_loop[curr_time] = station_locations[next_idx]
            friend_at_station_loop[curr_time] = True
            curr_idx = next_idx
    
    # Track visited states to avoid cycles
    visited_states = []  # [(location, time % loop_time, cost), ...]
    
    # Track best solution found so far to prune worse paths
    best_solution = None
    best_cost = float('inf')
    
    # 2D array to store paths efficiently
    paths = [[start]]  # First path with just the start location
    
    # Priority queue: (cost, time, location, path_id)
    pq = [(0, 0, start, 0)]  # path_id 0 is [start]
    
    # Count expanded states for safety
    expanded_states = 0
    
    while pq:
        cost, time, location, path_id = heapq.heappop(pq)
        expanded_states += 1
        
        # Prune path if we already have a better solution
        if cost >= best_cost:
            continue
        
        # State key represents our current position in space and the loop time
        time_in_loop = time % loop_time
        
        # Check if we've already found a better cost path to this state
        better_path_exists = False
        for loc, t_loop, c in visited_states:
            if loc == location and t_loop == time_in_loop and c <= cost:
                better_path_exists = True
                break
        
        if better_path_exists:
            continue
            
        # Remove any existing state for this location and time that has worse cost
        new_visited_states = []
        for loc, t_loop, c in visited_states:
            if not (loc == location and t_loop == time_in_loop):
                new_visited_states.append((loc, t_loop, c))
        new_visited_states.append((location, time_in_loop, cost))
        visited_states = new_visited_states
        
        # Check if we can intercept at this location and time
        if location in station_locations and friend_at_station_loop[time_in_loop]:
            if friend_positions_loop[time_in_loop] == location:
                # We found an interception point - check if it's the best solution
                if cost < best_cost:
                    best_cost = cost
                    best_solution = (cost, time, paths[path_id])
                elif cost == best_cost and (best_solution is None or time < best_solution[1]):
                    # Same cost but less time
                    best_solution = (cost, time, paths[path_id])
                # Continue exploring in case there's a better path
        
        # Explore neighbors - but only if there's potential for a better solution
        for neighbor, edge_cost, edge_time in graph[location]:
            new_cost = cost + edge_cost
            
            # Prune if this path is already worse than our best solution
            if new_cost >= best_cost:
                continue
            
            new_time = time + edge_time
            
            # Check if this state has been visited in a more efficient way
            new_time_in_loop = new_time % loop_time
            better_path_exists = False
            for loc, t_loop, c in visited_states:
                if loc == neighbor and t_loop == new_time_in_loop and c <= new_cost:
                    better_path_exists = True
                    break
            
            if better_path_exists:
                continue
            
            # Create a new path by extending the current path
            new_path = paths[path_id] + [neighbor]
            new_path_id = len(paths)
            paths.append(new_path)
            
            # Add this path to the priority queue with just the path ID instead of the full path
            heapq.heappush(pq, (new_cost, new_time, neighbor, new_path_id))
    
    # Return the best solution found, or None if interception is impossible
    return best_solution

# Test the function with the provided examples
if __name__ == "__main__":
    # Example 1, Simple
    roads_ex1 = [(6,0,3,1), (6,7,4,3), (6,5,6,2), (5,7,10,5), (4,8,8,5), (5,4,8,2),
    (8,9,1,2), (7,8,1,3), (8,3,2,3), (1,10,5,4), (0,1,10,3), (10,2,7,2),
    (3,2,15,2), (9,3,2,2), (2,4,10,5)]
    stations_ex1 = [(0,1), (5,1), (4,1), (3,1), (2,1), (1,1)]
    start_ex1 = 6
    friendStart_ex1 = 0
    
    result_ex1 = intercept(roads_ex1, stations_ex1, start_ex1, friendStart_ex1)
    print("Example 1 Result:", result_ex1)
    print("Example 1 Expected: (7, 9, [6,7,8,3])")
    print("Example 1 Matches expected:", result_ex1 == (7, 9, [6,7,8,3]))
    print("\n")
    
    # Example 2, Unsolvable
    roads_ex2 = [(0,1,35,3), (1,2,5,2), (2,0,35,4), (0,4,10,1), (4,1,22,2),
    (1,5,65,1), (5,2,70,1), (2,3,10,1), (3,0,20,3)]
    stations_ex2 = [(4,3), (5,2), (3,4)]
    start_ex2 = 0
    friendStart_ex2 = 4
    
    result_ex2 = intercept(roads_ex2, stations_ex2, start_ex2, friendStart_ex2)
    print("Example 2 Result:", result_ex2)
    print("Example 2 Expected: None")
    print("Example 2 Matches expected:", result_ex2 is None)
    
    # Example 3, Repeated Locations
    roads_ex3 = [(0,1,35,7), (1,2,5,4), (2,0,35,6), (0,4,10,5), (4,1,22,3),
    (1,5,60,4), (5,3,70,2), (3,0,10,7)]
    stations_ex3 = [(4,2), (5,1), (3,4)]
    start_ex3 = 0
    friendStart_ex3 = 3
    
    result_ex3 = intercept(roads_ex3, stations_ex3, start_ex3, friendStart_ex3)
    print("Example 3 Result:", result_ex3)
    print("Example 3 Expected: (160, 39, [0,1,2,0,1,2,0,4])")
    print("Example 3 Matches expected:", result_ex3 == (160, 39, [0,1,2,0,1,2,0,4]))
    
    # Example 4, Multiple routes with same cost but different total time
    roads_ex4 = [(0,1,10,7), (0,2,10,3), (2,0,1,4), (1,0,1,7)]
    stations_ex4 = [(2,4), (1,3)]
    start_ex4 = 0
    friendStart_ex4 = 1
    
    result_ex4 = intercept(roads_ex4, stations_ex4, start_ex4, friendStart_ex4)
    print("\nExample 4 Result:", result_ex4)
    print("Example 4 Expected: (10, 3, [0,2])")
    print("Example 4 Matches expected:", result_ex4 == (10, 3, [0,2]))