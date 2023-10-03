import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

VELOCITY = 8
LENGTH = 10.668
WIDTH = 3.048
REFRESH_RATE = 0.5 #in seconds
# REFRESH_RATE = 0.01 #in seconds
IDEAL_REFRESH = .01
RADIUS = 18 # 18 meters

class Car():
    def __init__(self, x_init, y_init, alpha_init):
            
        self.x_init = x_init
        self.y_init = y_init
        self.alpha_init = alpha_init

        self.x_hist = [x_init]
        self.y_hist = [y_init]
        self.theta_hist = [0]

    def drive(self, duration, alpha=0):
        for _ in range(int(duration/REFRESH_RATE)):
            omega:float = VELOCITY / LENGTH * np.tan(alpha)
            theta_next = self.theta_hist[-1] + omega * REFRESH_RATE
            self.theta_hist.append(theta_next)

            x_vel = VELOCITY * np.sin(self.theta_hist[-1])
            y_vel = VELOCITY * np.cos(self.theta_hist[-1])

            x_next = self.x_hist[-1] + x_vel * REFRESH_RATE
            y_next = self.y_hist[-1] + y_vel * REFRESH_RATE

            self.x_hist.append(x_next)
            self.y_hist.append(y_next)

    def getPath(self, radius, start, end):
        # Calculate circle
        cir = 2 * np.pi * radius
        theta = end - start
        # Divide into frame-long arcs
        dist = cir * theta / (2*np.pi)
        segments = np.floor(dist / VELOCITY)
        # Get array of ideal points

        # Find theta for each of the points

        # Return theta array


    def plot(self):
        fig, ax = plt.subplots()

        #Plot the boundry circle 
        circle = patches.Circle((0, 0), radius=RADIUS, fill=False, edgecolor='r', linewidth=3, linestyle='dotted')
        ax.add_patch(circle)
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_aspect('equal', 'box')

        #Plot origin point
        plt.scatter(0, 0, marker='x', color='red')  

        plt.plot(self.x_hist, self.y_hist, linewidth=1)
        plt.show()


if __name__ == '__main__':
    # Part 1 
    p1Car = Car(0, 0, 0)

    # Get to edge of circle
    p1Car.getPath(())
    # Loop through thetas on the path and drive for 1 frame each
    p1Car.drive((np.pi * 9) / VELOCITY, np.arctan(LENGTH / 9))
    # Drive along circle

    p1Car.drive(15, np.arctan(LENGTH / RADIUS))

    p1Car.plot()

    # Part 2 - Differential Steering
    # L:R power ratio = (r + (L/2)) / (r - (L/2))
    p2Car = (Car(0, 0, 0))
    
    # Get to edge of cirlce
    # r = 9, L = 3.048
    # R:L ratio = (9 + (3.048/2)) / ((9 - (3.048/2)) = 1.4077046548956662
    # R power set to 100 and L power set to 140.78 for X seconds

    # Drive along circle
    # R:L ratio = (18 + (3.048/2)) / ((18 - (3.048/2)) = 1.184996358339403
    #R power set to 100 and L power set to 118.50 for 2X seconds


    #Part 3 - 
    
