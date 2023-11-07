import heapq
import math
import numpy as np
import matplotlib.pyplot as plt
import time

START_COORDS = (0, 7)
END_COORDS = (6, 2)

STARTING_ALT = 0
ENDING_ALT = 0

N = (0, 1)
NE = (1, 1)
E = (1, 0)
SE = (1, -1)
S = (0, -1)
SW = (-1, -1)
W = (-1, 0)
NW = (-1, 1)

show_animation = False

sqrt_2 = math.sqrt(2)
directions_cost_map = {N: 1, NE: sqrt_2, E: 1, SE: sqrt_2, S: 1, SW: sqrt_2, W: 1, NW: sqrt_2}

terrain_map = [
    [4, 3, 2, 2, 2, 2, 1, 1, 1, 1],
    [3, 2, 2, -1, 2, 1, 1, 1, 2, 1],
    [3, 2, 2, -1, 2, 1, ENDING_ALT, 3, 3, 1],
    [2, 2, 1, -1, 4, 2, 4, 4, 3, 1],
    [2, 1, 1, -1, 4, 4, 4, 4, 2, 1],
    [1, 1, 1, -1, 3, 4, 3, 2, 2, 1],
    [1, 2, 2, 2, 2, 3, 1, 1, 1, 1],
    [STARTING_ALT, 1, 1, 1, 1, 1, 1, 1, 2, 2],
]
class Node:
    def __init__(self, x, y, parent):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost_g = 0
        self.cost_t = 0
        self.cost_h = 0
        self.cost_f = 0

    def __str__(self):
        return f"({self.x}, {self.y})"
    
    
    def __lt__(self, comparison_node):
        return self.cost_f < comparison_node.cost_f

class AStarPlanner:
    def __init__(self, start_node, end_node, terrain_map):
        self.start_node = start_node
        self.end_node = end_node
        self.terrain_map = terrain_map

    def find_path(self):
        open_list = []
        closeList = set()

        heapq.heappush(open_list, self.start_node) 

        while open_list:
            current_node = heapq.heappop(open_list)
            closeList.add((current_node.x, current_node.y))

            if show_animation:  # pragma: no cover
                plt.plot(current_node.x,
                         current_node.y, "xc")
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                plt.pause(0.5)

            if current_node.x == self.end_node.x and current_node.y == self.end_node.y:
                # Path found, backtrack to get path
                path = []
                while current_node.parent != None:
                    path.append(current_node)
                    current_node = current_node.parent
                path.append(start_node)
                return path

            # Generate children nodes
            for dir, cost in directions_cost_map.items():
                child_x = current_node.x + dir[0]
                child_y = current_node.y + dir[1]

                # Check that this location is in the defined area
                if 0 > child_x or child_x > 9:
                    print("ending because the element was found to be outside of x boundries")
                    continue
                if 0 > child_y or child_y > 7:
                    print("ending because the element was found to be outside of y boundries")
                    continue

                # Check that this location isn't in lake
                if terrain_map[child_y][child_x] == -1:
                    print("ending because the element was found to be in the lake")
                    continue

                # Check that its not in the closed set
                if (child_x, child_y) in closeList:
                    print("ending because the element was found inside closeList")
                    continue

                child_node = Node(child_x, child_y, current_node)
                child_node.parent = current_node
                child_node.cost_g = current_node.cost_g + cost
                child_node.cost_h = self.calc_h_cost(child_node)
                child_node.cost_t = self.calc_t_cost(current_node, child_node)
                # child_node.cost_f = child_node.cost_g + child_node.cost_t + child_node.cost_h
                child_node.cost_f = child_node.cost_t + child_node.cost_h

                for n in open_list:
                    if child_node == n and child_node.cost_g > n.cost_g:
                        continue
                
                heapq.heappush(open_list, child_node)
        return None
        
    @staticmethod
    def calc_t_cost(current, child):
        #Uphill
        if terrain_map[current.y][current.x] < terrain_map[child.y][child.x]:
            return 3
        elif terrain_map[current.y][current.x] > terrain_map[child.y][child.x]:
            return 1.5
        else:
            return 2
        
    def calc_h_cost(self, node):
        w = 0.5  # weight of heuristic
        return w * math.sqrt((node.x - self.end_node.x)**2 + (node.y - self.end_node.y)**2)

    

if __name__ == "__main__":
    start_node = Node(START_COORDS[0], START_COORDS[1], None)
    end_node = Node(END_COORDS[0], END_COORDS[1], None)

    planner = AStarPlanner(start_node=start_node, end_node=end_node, terrain_map=terrain_map)
    fig, ax = plt.subplots()
    plt.imshow(np.array(terrain_map), cmap='terrain')
    plt.colorbar()

    ax.grid(True)
    ax.scatter(start_node.x, start_node.y, color='green', s=100)
    ax.scatter(end_node.x, end_node.y, color='blue', s=100)

    start_time = time.time()
    path = planner.find_path()
    end_time = time.time()

    time_elapsed = end_time - start_time
    print(f"time elapsed: {time_elapsed}")

    path_x = [node.x for node in path]
    path_y = [node.y for node in path]
    plt.plot(path_x, path_y, 'ro-')
    plt.show()

    for p in path:
        print(p)