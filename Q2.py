import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

VELOCITY = 8
LENGTH = 10.668
# REFRESH_RATE = 0.5 #in seconds
REFRESH_RATE = 0.01 #in seconds

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

    def plot(self):
        fig, ax = plt.subplots()

        #Plot the boundry circle 
        circle = patches.Circle((0, 0), radius=18, fill=False, edgecolor='r', linewidth=3, linestyle='dotted')
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

     p1Car.drive((np.pi * 9) / VELOCITY, np.arctan(LENGTH / 9))
     p1Car.drive(15, np.arctan(LENGTH / 18))

     p1Car.plot()