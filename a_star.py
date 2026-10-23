import numpy as np
import matplotlib.pyplot as plt
import heapq
from scipy.interpolate import CubicSpline
from scipy.interpolate import BSpline

class A_star:
    def __init__(self):
        self.map_size = [-30, 10]

        self.w = 2
        self.l = 3
        self.buffer = 0.75

        self.grid_size = 50
        self.num_obs = 50
        self.num_pat = 0 # 200

        self.map_grid_x = np.linspace(self.map_size[0], self.map_size[1], self.grid_size)
        self.map_grid_y = np.linspace(self.map_size[0], self.map_size[1], self.grid_size)

        self.create_obstacles(self.num_obs)

        self.create_grid_obstacles()
        self.create_grid_patches()
        self.start, self.goal = self.init_start_goal()

        self.parent = {}
        self.cost = {}

        self.parent[self.start] = None
        self.cost[self.start] = 0

        self.frontier = []
        self.path = []

        self.frontier_hist = [] # for plotting

    def create_quad_map_obstacle(self, l, w):

        x_obs = np.random.randint(self.map_size[0], self.map_size[1])
        y_obs = np.random.randint(self.map_size[0], self.map_size[1])

        obs_horizontal = [x_obs - w/2, x_obs + w/2, y_obs - l/2, y_obs + l/2]
        obs_vertical = [x_obs - l/2, x_obs + l/2, y_obs - w/2, y_obs + w/2]

        return [obs_horizontal, obs_vertical][np.random.choice([0, 1])]

    def create_obstacles(self, n):
        self.obstacles = []

        for i in range(n):
            obs = self.create_quad_map_obstacle(self.w, self.l)
            self.obstacles.append(obs)

    def check_obstacle(self, x, y):
        for obs in self.obstacles:
            if x >= obs[0] - self.buffer and x <= obs[1] + self.buffer and y >= obs[2] - self.buffer and y <= obs[3] + self.buffer:
                return True

    def check_borders(self, x, y):
        if x >= self.map_size[0] + self.buffer and x <= self.map_size[1] - self.buffer and y >= self.map_size[0] + self.buffer and y <= self.map_size[1] - self.buffer:
            return True

    def create_grid_obstacles(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = self.map_grid_x[i]
                y = self.map_grid_y[j]
                if self.check_obstacle(x, y):
                    self.grid[j][i] = 1

    # def create_grid_obstacles(self):
    #     self.grid = np.zeros((self.grid_size, self.grid_size))
    #     self.x_obs = np.random.randint(self.grid_size, size=self.num_obs)
    #     self.y_obs = np.random.randint(self.grid_size, size=self.num_obs)

    #     for k in range(self.num_obs):
    #             i = self.y_obs[k]
    #             j = self.x_obs[k]
    #             self.grid[i][j] = 1

    def create_grid_patches(self):
        self.x_pat = np.random.randint(self.grid_size, size=self.num_pat)
        self.y_pat = np.random.randint(self.grid_size, size=self.num_pat)

        for k in range(self.num_pat):
                i = self.y_pat[k]
                j = self.x_pat[k]
                self.grid[i][j] = 0.25

    def init_start_goal(self):
        x_start = np.random.randint(self.grid_size)
        y_start = np.random.randint(self.grid_size)

        while self.grid[y_start, x_start] == 1:
            x_start = np.random.randint(self.grid_size)
            y_start = np.random.randint(self.grid_size)

        x_goal = np.random.randint(self.grid_size)
        y_goal = np.random.randint(self.grid_size)

        while self.grid[y_goal, x_goal] == 1:
            x_goal = np.random.randint(self.grid_size)
            y_goal = np.random.randint(self.grid_size)

        map_start_x = x_start * (self.map_size[1] - self.map_size[0]) / self.grid_size + self.map_size[0]
        map_start_y = y_start * (self.map_size[1] - self.map_size[0]) / self.grid_size + self.map_size[0]
        map_goal_x = x_goal * (self.map_size[1] - self.map_size[0]) / self.grid_size + self.map_size[0]
        map_goal_y = y_goal * (self.map_size[1] - self.map_size[0]) / self.grid_size + self.map_size[0]

        self.map_start = [map_start_x, map_start_y]
        self.map_goal = [map_goal_x, map_goal_y]

        return tuple([x_start, y_start]), tuple([x_goal, y_goal])
    
    def neighbors(self, node):

        x, y = node[0], node[1]
        neighbors = []
        pairs = [tuple([x - 1, y]), tuple([x, y - 1]),
                 tuple([x, y + 1]), tuple([x + 1, y]),
                 tuple([x - 1, y - 1]), tuple([x - 1, y + 1]),
                 tuple([x + 1, y - 1]), tuple([x + 1, y + 1])]

        for k in range(8):
            # print(pairs[k])
            x = self.map_grid_x[pairs[k][0] - 1]
            y = self.map_grid_y[pairs[k][1] - 1]
            # if (pairs[k][0] >= 0 and pairs[k][0] < self.grid_size) and (pairs[k][1] >= 0 and pairs[k][1] < self.grid_size):
            if self.check_borders(x, y):
                if self.grid[pairs[k][1], pairs[k][0]] != 1:
                    neighbors.append(tuple(pairs[k]))

        return neighbors

    def a_star_search(self):
        cost = self.heuristic(self.start, self.goal)
        heapq.heappush(self.frontier, (cost, self.start))
        
        while self.frontier:
            current_cost, node = heapq.heappop(self.frontier)

            if node == self.goal:
                print("FOUND")
                return
            
            for neighbor in self.neighbors(node):
                g = self.cost[node] + 10 * self.turn_cost(node, neighbor) + self.heuristic(neighbor, self.goal)
                # print(f"cost: {g}")
                if neighbor not in self.cost or g < self.cost[neighbor]:
                    self.cost[neighbor] = g
                    self.parent[neighbor] = node
                    heapq.heappush(self.frontier, (g, neighbor))

                    self.frontier_hist.append([node])

        print("NO PATH")
        return

    def build_path(self):
        current = self.goal
        while current != self.start:
            self.path.append(current)
            current = self.parent[current]
        self.path.append(self.start)
        self.path.reverse()

    def turn_cost(self, node1, node2):
        if node1 == self.start:
            return 0

        
        x1, y1 = self.parent[node1]
        x2, y2 = node1
        x3, y3 = node2

        dx1, dy1 = x2 - x1, y2 - y1
        dx2, dy2 = x3 - x2, y3 - y2

        if dx1 * dy2 - dx2 * dy1 != 0:
            if (self.step_cost(self.parent[node1], node1) > 0):
                return 1000 * self.step_cost(node1, node2)
            else:
                return 500
        else:
            return 0

    def step_cost(self, node1, node2):
        x1, y1 = node1
        x2, y2 = node2

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        return (dx == 1 and dy == 1)

    def edge_cost(self, node1, node2):
        x1, y1 = node1
        x2, y2 = node2
        if self.grid[y2, x2] == 0.25:
            return 3
        return 1

    def heuristic(self, node1, node2):
        x1, y1 = node1
        x2, y2 = node2

        diag = (x2 - x1) ** 2 + (y2 - y1) ** 2
        # manhattan = abs(x2 - x1) + abs(y2 - y1)

        return diag
        
    def plot_nodes(self, nodes):
        x = []
        y = []
        for node in nodes:
            x.append(node[0])
            y.append(node[1])

        return x, y

    def create_parametric_cubic_spline(self, x_wp, y_wp,):
        # creates functions x(s) and y(s) for parametric cubic splines
        s_wp = np.array([0])
        for i in range(1, len(x_wp)):
            d = np.sqrt((x_wp[i] - x_wp[i - 1]) ** 2 +  (y_wp[i] - y_wp[i - 1]) ** 2)
            s_wp = np.append(s_wp, s_wp[i - 1] + d)

        self.x_spline = CubicSpline(s_wp, x_wp)
        self.y_spline = CubicSpline(s_wp, y_wp)

        return s_wp


    def create_cubic_spline_path(self):
        x_wp = []
        y_wp = []

        for i in range(len(self.path)):
            x = self.path[i][0] * (self.map_size[1] - self.map_size[0]) / self.grid_size + self.map_size[0]
            y = self.path[i][1] * (self.map_size[1] - self.map_size[0]) / self.grid_size + self.map_size[0]

            x_wp.append(x)
            y_wp.append(y)

        s_wp = self.create_parametric_cubic_spline(x_wp, y_wp)
        s = np.linspace(s_wp[0], s_wp[-1], 1000)
        x_s = self.x_spline(s)
        y_s = self.y_spline(s)

        return x_wp, y_wp, x_s, y_s
    
    def create_minimal_cubic_spline_path(self):
        x_wp = [self.path[0][0] * (self.map_size[1] - self.map_size[0]) / self.grid_size + self.map_size[0]]
        y_wp = [self.path[0][1] * (self.map_size[1] - self.map_size[0]) / self.grid_size + self.map_size[0]]
        for i in range(1, len(self.path) - 1):
            x = self.path[i][0]
            y = self.path[i][1]

            x1 = self.path[i - 1][0]
            y1 = self.path[i - 1][1]
            x2 = self.path[i + 1][0]
            y2 = self.path[i + 1][1]

            if x1 != x2 and y1 != y2:
                x_wp.append(x * (self.map_size[1] - self.map_size[0]) / self.grid_size + self.map_size[0])
                y_wp.append(y * (self.map_size[1] - self.map_size[0]) / self.grid_size + self.map_size[0])

        x_wp.append(self.path[-1][0] * (self.map_size[1] - self.map_size[0]) / self.grid_size + self.map_size[0])  
        y_wp.append(self.path[-1][1] * (self.map_size[1] - self.map_size[0]) / self.grid_size + self.map_size[0])

        s_wp = self.create_parametric_cubic_spline(x_wp, y_wp)
        s = np.linspace(s_wp[0], s_wp[-1], 1000)
        x_s = self.x_spline(s)
        y_s = self.y_spline(s)

        return x_wp, y_wp, x_s, y_s
    
if __name__ == "__main__":
    A = A_star()
    A.a_star_search()
    A.build_path()

    x_wp, y_wp, x_s, y_s = A.create_cubic_spline_path()

    fig, ax = plt.subplots(1, 2)

    # ax[0].set_xticks(np.linspace(0, A.grid_size, A.grid_size))
    # ax[0].set_yticks(np.linspace(0, A.grid_size, A.grid_size))
    # ax[0].set_xticklabels([])
    # ax[0].set_yticklabels([])
    # ax[0].grid()

    ax[1].set_title('Map of Area with obstacles')
    ax[1].set_xlim(-10, 10)
    ax[1].set_ylim(-10, 10)
    # ax[1].grid()
    ax[1].axis('scaled')
    ax[1].set_xticks(np.linspace(A.map_size[0], A.map_size[1], 9))
    ax[1].set_yticks(np.linspace(A.map_size[0], A.map_size[1], 9))

    ax[1].plot(A.map_start[0], A.map_start[1], 'o', c='orange', markersize=5)
    ax[1].plot(A.map_goal[0], A.map_goal[1], 'o', c='green', markersize=5)

    for obs in A.obstacles:
        x1 = [obs[0], obs[0]]
        y1 = [obs[2], obs[3]]

        x2 = [obs[0], obs[1]]
        y2 = [obs[3], obs[3]]

        x3 = [obs[1], obs[1]]
        y3 = [obs[3], obs[2]]

        x4 = [obs[1], obs[0]]
        y4 = [obs[2], obs[2]]

        ax[1].plot(x1, y1, c='black')
        ax[1].plot(x2, y2, c='black')
        ax[1].plot(x3, y3, c='black')
        ax[1].plot(x4, y4, c='black')

    ax[0].set_title('Occupancy Grid with Buffers')
    ax[0].imshow(A.grid, cmap='binary', origin='lower')
    ax[0].plot(A.start[0], A.start[1], 'o', c='orange', markersize=5)
    ax[0].plot(A.goal[0], A.goal[1], 'o', c='green', markersize=5)

    # for nodes in A.frontier_hist:
    #     x, y = A.plot_nodes(nodes)
    #     # print(f"x: {x}, y: {y}")
    #     ax[0].plot(x, y, 'o', c='blue', markersize=3)
    #     plt.pause(1e-4)

    x_path = []
    y_path = []
    for pair in A.path:
        x_path.append(pair[0])
        y_path.append(pair[1])

    ax[0].plot(x_path, y_path, c='cyan', markersize=3, linewidth=3)

    # ax[1].plot(x_wp, y_wp, c='red', linewidth=8, alpha=0.4)
    ax[1].plot(x_s, y_s, c='blue', linewidth=2)

    plt.show()