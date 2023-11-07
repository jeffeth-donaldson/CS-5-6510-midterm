import heapq
import matplotlib.pyplot as plt

START_X = 10
START_Y = 10
END_X = 25
END_Y = 25

class Node:
    def __init__(self, x_pos, y_pos, cost=0, parent_index=None) -> None:
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.cost = cost  # g-cost
        self.parent_index = parent_index
        self.heuristic = 0  # h-cost
        self.total_cost = 0  # f-cost

    def __lt__(self, other):
        return self.total_cost < other.total_cost

class GridPlanner:
    def __init__(self, obstacles: list[Node], grid_size=(35, 35)):
        self.obstacles = obstacles
        self.grid_size = grid_size
        self.obstacle_map = self.populate_obstical_map(obstacles, self.grid_size[1])
    
        # print(self.obstacle_map)

    @staticmethod
    def populate_obstical_map(obstacles, grid_size):
        map = [[False for _ in range(grid_size)] for _ in range(grid_size)]
        for obstacle_node in obstacles:
            map[obstacle_node.x_pos - 1][ obstacle_node.y_pos - 1] = True
        return map

    def heuristic(self, node1: Node, node2: Node):
        return abs(node1.x_pos - node2.x_pos) + abs(node1.y_pos - node2.y_pos)
    
    def plot_map(self):
        fig, ax = plt.subplots()

        for obstacle in self.obstacles:
            ax.scatter(obstacle.x_pos, obstacle.y_pos, color='red', s=100)
        ax.grid(True)
        ax.scatter(START_X, START_Y, color='green', s=100)
        ax.scatter(END_X, END_Y, color='blue', s=100)
    
    def plot_path(self, path):
        fig, ax = plt.subplots()

        for obstacle in self.obstacles:
            ax.scatter(obstacle.x_pos, obstacle.y_pos, color='red', s=100)

        if path:
            x_coords = [x for x, y in path]
            y_coords = [y for x, y in path]
            plt.plot(x_coords, y_coords, color='black')
            ax.scatter(x_coords, y_coords, color='black', s=20)
            ax.scatter(x_coords[0], y_coords[0], color='green', s=100)
            ax.scatter(x_coords[-1], y_coords[-1], color='blue', s=100)

        ax.grid(True)
        plt.show()

    def a_star_plan(self, start: Node, end: Node):
        open_list = []
        closed_list = set()
        heapq.heappush(open_list, start)

        while open_list:
            current_node = heapq.heappop(open_list)
            closed_list.add((current_node.x_pos, current_node.y_pos))

            #live plotting
            # plt.plot(current_node.x_pos, current_node.y_pos, "xc")
            # plt.pause(0.0001)

            if current_node.x_pos == end.x_pos and current_node.y_pos == end.y_pos:
                final_cost = current_node.total_cost
                # print(f"cost: {current_node.cost}")
                # print(f"heuristic: {current_node.heuristic}")
                # print(f"total_cost: {current_node.total_cost}")
                path = []
                while current_node is not None:
                    path.append((current_node.x_pos, current_node.y_pos))
                    current_node = current_node.parent_index
                return path[::-1], final_cost

            for dx, dy in  [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                x, y = current_node.x_pos + dx, current_node.y_pos + dy

                if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                    if self.obstacle_map[x-1][y-1]:
                        continue

                    if (x, y) in closed_list:
                        continue

                    tentative_g_cost = current_node.cost + 1
                    neighbor = Node(x, y)
                    neighbor.parent_index = current_node

                    if neighbor not in open_list:
                        neighbor.cost = tentative_g_cost
                        neighbor.heuristic = self.heuristic(neighbor, end)
                        neighbor.total_cost = neighbor.cost + neighbor.heuristic
                        heapq.heappush(open_list, neighbor)

                    else:
                        if tentative_g_cost < neighbor.cost:
                            neighbor.cost = tentative_g_cost
                            neighbor.total_cost = neighbor.cost + neighbor.heuristic
                            neighbor.parent_index = current_node

        return None  # Path not found
    
def main():  
    obstacle_nodes = []
    #Borders
    for i in range(0, 36):
        obstacle_nodes.append(Node(35, i))
        obstacle_nodes.append(Node(i, 35))
        obstacle_nodes.append(Node(0, i))
        obstacle_nodes.append(Node(i, 0))
    # Inner Obstacles
    for i in range(0, 20):
        obstacle_nodes.append(Node(20, 35-i))
    for i in range(0, 25):
        obstacle_nodes.append(Node(15, i))

    grid_planner = GridPlanner(obstacles=obstacle_nodes)
    # grid_planner.plot_map()
    start_node = Node(START_X, START_Y)
    end_node = Node(END_X, END_Y)

    path, cost = grid_planner.a_star_plan(start_node, end_node)
    # grid_planner.plot_path(path)

    return cost

if __name__ == '__main__':
    main()