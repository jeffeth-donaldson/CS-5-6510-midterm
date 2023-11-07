import timeit

from our_aStar import main as our_aStar_test
from their_aStar import main as their_aStar_test
from djikstra import main as djikstra_test
from bidirectional_aStar import main as bidirectional_aStar_test
from breathFirstSearch import main as breathFirstSearch_test
from rrt_star import main as rrt_star_test

class PathfindingAlgorithm:
    def __init__(self, name, test_func):
        self.name = name
        self.test_func = test_func
        self.costs= []
        self.avg_costs = None
        self.avg_time = None

algo_list = [
    PathfindingAlgorithm('Our A-star', our_aStar_test),
    PathfindingAlgorithm('Their A-star', their_aStar_test),
    PathfindingAlgorithm('Djikstra', djikstra_test),
    PathfindingAlgorithm('Bidirectional A-Star', bidirectional_aStar_test),
    PathfindingAlgorithm('Breath-first Search', breathFirstSearch_test),
    PathfindingAlgorithm('RRT', rrt_star_test)
]

def results_wrapper(test_func, cost_list):
    cost_list.append(test_func())


def main():
    for algo in algo_list:
        algo.avg_time = timeit.timeit(lambda: results_wrapper(algo.test_func, algo.costs), number=5)
        algo.avg_costs = sum(algo.costs) / len(algo.costs)
    for algo in algo_list:
        print(f"{algo.name}\n\t avg time: {algo.avg_time} \n\t avg cost: {algo.avg_costs}\n")


if __name__ == "__main__":
    main()