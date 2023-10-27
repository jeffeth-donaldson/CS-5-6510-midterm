import heapq
import matplotlib.pyplot as plt

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
    def __init__(self, obstacles: list[Node], grid_size=(75, 75)):
        self.obstacles = obstacles
        self.grid_size = grid_size

    def heuristic(self, node1: Node, node2: Node):
        return abs(node1.x_pos - node2.x_pos) + abs(node1.y_pos - node2.y_pos)
    
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

            if current_node.x_pos == end.x_pos and current_node.y_pos == end.y_pos:
                path = []
                while current_node is not None:
                    path.append((current_node.x_pos, current_node.y_pos))
                    current_node = current_node.parent_index
                return path[::-1]

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                x, y = current_node.x_pos + dx, current_node.y_pos + dy

                if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                    if any(obstacle.x_pos == x and obstacle.y_pos == y for obstacle in self.obstacles):
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
    for i in range(0, 75):
        obstacle_nodes.append(Node(75, i))
        obstacle_nodes.append(Node(i, 75))
        obstacle_nodes.append(Node(0, i))
        obstacle_nodes.append(Node(i, 0))
    #Inner Obstacles
    for i in range(0, 60):
        obstacle_nodes.append(Node(20, i))
    for i in range(0, 60):
        obstacle_nodes.append(Node(40, 75 - i))

    grid_planner = GridPlanner(obstacles=obstacle_nodes)
    start_node = Node(5, 5)
    end_node = Node(70, 70)

    path = grid_planner.a_star_plan(start_node, end_node)
    print(f"Path found: {path}")
    grid_planner.plot_path(path)

if __name__ == '__main__':
    main()