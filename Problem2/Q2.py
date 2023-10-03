import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

VELOCITY = 8
LENGTH = 10.668 # meters
WIDTH = 3.048 #meters
REFRESH_RATE = 0.01 #in seconds
# REFRESH_RATE = 0.01 #in seconds
LENGTH = 10.668
WIDTH = 3.048
# REFRESH_RATE = 0.5 #in seconds
REFRESH_RATE = 0.01 #in seconds
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

    def diferential_drive(self, duration, turning_radius):
        right_velocity = (VELOCITY / turning_radius)  * (turning_radius - WIDTH / 2)
        left_velocity = (VELOCITY / turning_radius)  * (turning_radius + WIDTH / 2)
        omega_dot = (right_velocity - left_velocity) / WIDTH
        omega = self.theta_hist[-1]
        for _ in range(int(duration/REFRESH_RATE)):
            # omega += omega_dot * REFRESH_RATE
            theta_next = self.theta_hist[-1] + omega_dot * REFRESH_RATE
            self.theta_hist.append(theta_next)

            x_vel = -((right_velocity + left_velocity)/2) * np.sin(self.theta_hist[-1])
            y_vel = ((right_velocity + left_velocity)/2) * np.cos(self.theta_hist[-1])

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
        plt.title("Motion of Ackerman Robot")
        plt.xlabel("X position in meters from origin")
        plt.ylabel("Y position in meters from origin")
        plt.show()

    def plot_setup(self):
        fig, ax = plt.subplots()

        #Plot the boundry circle 
        circle = patches.Circle((0, 0), radius=RADIUS, fill=False, edgecolor='r', linewidth=3, linestyle='dotted')
        ax.add_patch(circle)
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_aspect('equal', 'box')

        #Plot origin point
        plt.scatter(0, 0, marker='x', color='red')  
    def plot_no_graph(self, label):
        plt.plot(self.x_hist, self.y_hist, linewidth=1, label=label)
        plt.title("Motion of Skid-Steer Robot")
        plt.xlabel("X position in meters from origin")
        plt.ylabel("Y position in meters from origin")

if __name__ == '__main__':
    # Part 1 
    # p1Car = Car(0, 0, 0)

    # # Get to edge of circle
    # # p1Car.getPath(())
    # # Loop through thetas on the path and drive for 1 frame each
    # p1Car.drive((np.pi * 9) / VELOCITY, np.arctan(LENGTH / 9))
    # # Drive along circle

    # p1Car.drive(15, np.arctan(LENGTH / RADIUS))
    # Initialize Car Location
    # p1Car = Car(0, 0, 0)
    # angular_velocities = []
    # time = []
    # # Calculate Radius of circle adjusted for width of car
    # ADJUSTED_RADIUS = np.floor(RADIUS - (WIDTH/2))
    # HALF_CIRCLE_LENGTH = np.pi * (ADJUSTED_RADIUS / 2)
    # print("HALF_CIRCLE_LENGTH / VELOCITY",HALF_CIRCLE_LENGTH / VELOCITY)
    # for i in range(int(HALF_CIRCLE_LENGTH / VELOCITY)):
    #     angular_velocities.append(VELOCITY/(ADJUSTED_RADIUS/2))
    #     time.append(i)
    # end = time[-1]
    # print("First alpha", np.arctan(LENGTH / (ADJUSTED_RADIUS/2)))
    # p1Car.drive(HALF_CIRCLE_LENGTH / VELOCITY, np.arctan(LENGTH / (ADJUSTED_RADIUS/2)))
    # CIRCUMFERENCE = 2* np.pi * ADJUSTED_RADIUS 
    # print("Second Time:",CIRCUMFERENCE/VELOCITY)
    # for i in range(int(CIRCUMFERENCE/VELOCITY)):
    #     angular_velocities.append(VELOCITY/ADJUSTED_RADIUS)
    #     time.append(end+i)
    # print("Second alpha",np.arctan(LENGTH / ADJUSTED_RADIUS))   
    # p1Car.drive(CIRCUMFERENCE/VELOCITY, np.arctan(LENGTH / ADJUSTED_RADIUS))

    # plt.plot(time, angular_velocities)
    # plt.title('Angular Velocity of Ackermann Robot')
    # plt.xlabel('Time in seconds')
    # plt.ylabel('Angular Velocity in Radians/Second')
    # plt.show()

    # p1Car.plot()

    # Part 2 - Differential Steering
    # L:R power ratio = (r + (L/2)) / (r - (L/2))
    # p2Car = (Car(0, 0, 0))
    
    # Get to edge of cirlce
    # r = 9, L = 3.048
    # R:L ratio = (9 + (3.048/2)) / ((9 - (3.048/2)) = 1.4077046548956662
    # R power set to 100 and L power set to 140.78 for X seconds
    p2Car.diferential_drive((np.pi * ADJUSTED_RADIUS / 2) / VELOCITY, ADJUSTED_RADIUS / 2)

    # Drive along circle
    # R:L ratio = (18 + (3.048/2)) / ((18 - (3.048/2)) = 1.184996358339403
    #R power set to 100 and L power set to 118.50 for 2X seconds
    p2Car.diferential_drive((2 * (np.pi * ADJUSTED_RADIUS) / VELOCITY), ADJUSTED_RADIUS)
    p2Car.plot()

    #Part 3 - 
    RADIUS = 9 #meters
    Car(RADIUS, 0, 0).plot_setup()
    for delta in [1, 0.1, 0.01]:
        REFRESH_RATE = delta
        p3Car = Car(RADIUS, 0, 0)
        p3Car.diferential_drive(2*RADIUS*np.pi/VELOCITY, -RADIUS)

        p3Car.plot_no_graph(f'{1/delta}Hz')
    plt.legend()
    plt.show()
    
