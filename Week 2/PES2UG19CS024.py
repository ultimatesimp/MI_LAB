
from queue import PriorityQueue


def A_star_Traversal(cost, heuristic, start_point, goals):
    
    path = dict()
    explored = set()
    state = []
    frontier = PriorityQueue()
    global last

    frontier.put((heuristic[start_point] + 0, start_point, 0, -1))
    
    while not frontier.empty():

        state = frontier.get()
        
        if state[1] not in explored:
            path[state[1]] = state[3]
            explored.add(state[1])

        
        if state[1] in goals:

            new_path = [state[1]]
            pointer = state[1]
            while pointer != start_point:
                new_path.append(path[pointer])
                pointer = path[pointer]
                
            break

        for i in range(len(cost[state[1]])):
            if cost[state[1]][i] > 0 and i not in explored:
                path_cost = cost[state[1]][i] + state[2] # calculate g(n) or total path cost
                frontier.put((heuristic[i] + path_cost, i, path_cost, state[1]))

    new_new_path = [i for i in new_path[::-1]]
    

    return new_new_path

    
def DFS_Traversal(cost, start_point, goals):

    #initialize explored and path
    explored = set()
    path = []

    # Perform DFS to get path
    path = dfs_helper(cost, start_point, goals, path, explored)

    return path



def dfs_helper(cost, node, goals, path, explored):
    
    # add node to explored and path
    explored.add(node)
    path.append(node)
    
    
    if node in goals: # check if node in goals
        return path
    else:
        for i in range(len(cost[node])):
            if cost[node][i] > 0 and i not in explored: # look for nodes that have a cost greater than 0 and are not in explored
                return dfs_helper(cost, i, goals, path, explored) # call dfs helper function
    
    

