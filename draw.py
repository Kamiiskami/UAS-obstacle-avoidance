import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

# draw obstacles on the map  - move to another file
class Draw:
    ###################### draw rect ################################################################
    def drawrect(self, ax, rect):
        coord = rect
        coord.append(coord[0])  # repeat the first point to create a 'closed loop'
        xs, ys = zip(*coord)  # create lists of x and y values
        plt.plot(xs, ys)
        # ax.add_patch(Rectangle((someX - dx, someY - dy), 2, 2, alpha=1))

    ################# draw circle ###################################################################
    def drawcir(self, ax, cir, r):
        # circle1 = plt.Circle(cir, r, linewidth=5, color = 'dodgerblue', fill = None)
        # ax.add_artist(circle1)
        circle2 = plt.Circle(cir, r, color='dodgerblue')
        ax.add_artist(circle2)


def findaction(point1, matrix):
    pp1 = np.unravel_index(point1, matrix)
    x1 = pp1[1]
    y1 = pp1[0]
    action = np.zeros(len(x1)-1)
    deltax = np.zeros(len(x1)-1)
    deltay = np.zeros(len(x1) - 1)
    for i in range(len(x1)-1):
        deltax[i] = pp1[1][i+1] - pp1[1][i]
        deltay[i] = pp1[0][i+1] - pp1[0][i]
        if deltax[i] == 0 and deltay[i] < 0:
            action[i] = 0 #up
        if deltax[i] == 0 and deltay[i] > 0:
            action[i] = 1  # down
        if deltax[i] > 0 and deltay[i] == 0:
            action[i] = 2 #right
        if deltax[i] < 0 and deltay[i] == 0:
            action[i] = 3 # left
        if deltax[i] > 0 and deltay[i] < 0:
            action[i] = 4 #UP_RIGHT
        if deltax[i] < 0 and deltay[i] < 0:
            action[i] = 5 #UP_LEFT
        if deltax[i] > 0 and deltay[i] > 0:
            action[i] = 6 #DOWN_RIGHT
        if deltax[i] < 0 and deltay[i] > 0:
            action[i] = 7 #DOWN_LEFT

    return action

def drawbound(x1, y1, action, ax):
    for i in range(len(action)):
        #y1[i] = 49 - y1[i]
        if action[i] == 0: #up
            Draw().drawcir(ax, (x1[i], y1[i]), 1.7/4)
            Draw().drawcir(ax, (x1[i], y1[i] - 68.6/4), 1.7/4)
            plt.fill([x1[i]-1.7/4, x1[i]-1.7/4, x1[i]+1.7/4, x1[i]+1.7/4], [y1[i], y1[i]- 68.6/4, y1[i]- 68.6/4, y1[i]], color='lightskyblue')
            print("upp")
        if action[i] == 1: # down
            Draw().drawcir(ax, (x1[i], y1[i]), 1.7/4)
            Draw().drawcir(ax, (x1[i], y1[i] + 31.9/4), 1.7/4)
            plt.fill([x1[i] - 1.7 / 4, x1[i] - 1.7 / 4, x1[i] + 1.7 / 4, x1[i] + 1.7 / 4],
                     [y1[i], y1[i]+ 31.9/4, y1[i]+ 31.9/4, y1[i]], color='lightskyblue')
            print("dwon")
        if action[i] == 2: # right
            Draw().drawcir(ax, (x1[i], y1[i]), 7.1/4)
            Draw().drawcir(ax, (x1[i]+49.3/4, y1[i]), 7.1/4)
            plt.fill([x1[i], x1[i]+49.3/4, x1[i]+49.3/4, x1[i]],[y1[i]- 7.1/4, y1[i]- 7.1/4, y1[i]+ 7.1/4, y1[i] + 7.1/4], color = 'lightskyblue')
            print("right")
        if action[i] == 3:  # left
            Draw().drawcir(ax, (x1[i], y1[i]), 7.1 / 4)
            Draw().drawcir(ax, (x1[i] - 49.0 / 4, y1[i]), 7.1 / 4)
            plt.fill([x1[i], x1[i] - 49.0 / 4, x1[i] - 49.0 / 4, x1[i]], [y1[i] - 7.1 / 4, y1[i] - 7.1 / 4, y1[i] + 7.1 / 4, y1[i] + 7.1 / 4], color='lightskyblue')
            print("left")
        if action[i] == 4: #UP_RIGHT
            Draw().drawcir(ax, (x1[i], y1[i]), 1.5/4)
            Draw().drawcir(ax, (x1[i] + 68.6/4, y1[i] - 68.6/4), 1.7/4)
            plt.fill([x1[i]-1.7/4, x1[i]-1.7/4, x1[i]+1.7/4, x1[i]+1.7/4], [y1[i], y1[i]- 68.6/4, y1[i]- 68.6/4, y1[i]], color='lightskyblue')
            print("up_right")
        if action[i] == 5: #UP_LEFT
            Draw().drawcir(ax, (x1[i]-0.5, y1[i]-0.5), 1.5/4)
            Draw().drawcir(ax, (x1[i]-0.5 - 68.6/4, y1[i]-0.5- 68.6/4), 1.7/4)
            plt.fill([x1[i]-1.7/4-0.5, x1[i]-1.7/4-0.5, x1[i]+1.7/4-0.5, x1[i]+1.7/4-0.5], [y1[i]-0.5, y1[i]- 68.6/4-0.5, y1[i]- 68.6/4-0.5, y1[i]-0.5], color='lightskyblue')
            print("up_left")
        elif action[i] == 6: #DOWN_RIGHT
            Draw().drawcir(ax, (x1[i], y1[i]), 1.5/4)
            Draw().drawcir(ax, (x1[i]+ 68.6/4, y1[i] + 68.6/4), 1.7/4)
            plt.fill([x1[i]-1.7/4, x1[i]-1.7/4, x1[i]+ 68.6/4, x1[i]+ 68.6/4], [y1[i], y1[i] + 68.6/4, y1[i] + 68.6/4, y1[i]], color='lightskyblue')
            print("down_right")
        elif action[i] == 7: #DOWN_LEFT
            Draw().drawcir(ax, (x1[i]-0.5, y1[i]-0.5), 1.5/4)
            Draw().drawcir(ax, (x1[i]-0.5 - 68.6/4, y1[i]-0.5 - 68.6/4), 1.7/4)
            plt.fill([x1[i]-1.7/4-0.5, x1[i]-1.7/4-0.5, x1[i]+1.7/4-0.5, x1[i]+1.7/4], [y1[i]-0.5, y1[i]- 68.6/4-0.5, y1[i]- 68.6/4-0.5, y1[i]-0.5], color='lightskyblue')
            print("down_left")


def trajectory(point1, matrix, poly1, poly2, poly3, poly4, poly5):
    cmap = colors.ListedColormap(['white', 'red', 'black'])
    data = np.zeros(matrix)
    for i in range(len(point1)):
        ax = plt.gca()
        data[np.unravel_index(point1[i], matrix)] = 1
        data[poly1[0]:poly1[1] + 1, poly1[2]:poly1[3] + 1] = 2
        data[poly2[0]:poly2[1] + 1, poly2[2]:poly2[3] + 1] = 2
        data[poly3[0]:poly3[1] + 1, poly3[2]:poly3[3] + 1] = 2
        data[poly4[0]:poly4[1] + 1, poly4[2]:poly4[3] + 1] = 2
        data[poly5[0]:poly5[1] + 1, poly5[2]:poly5[3] + 1] = 2

        ax.grid(which='minor', axis='both', linestyle=':', color='k', linewidth=0.3)  # shows grid lines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.6)
        major_ticks = np.arange(-.5, matrix[0], 2)
        minor_ticks = np.arange(-.5, matrix[1], 2)

        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True) 
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.imshow(data, cmap=cmap)

    pp1 = np.unravel_index(point1, matrix)
    x1 = pp1[1]
    y1 = pp1[0]
    action1 = findaction(point1, matrix)

    # plot path
    plt.plot(x1, y1, color='red')
    plt.plot(x1, y1, '--', linewidth=1.5, color='k')
    drawbound(x1, y1, action1, ax)

    plt.plot([40], [0], 'P', color='blue')
    plt.plot([0], [45], 'ob', color='blue')
    plt.legend(('Waypoint', 'Trajectory', 'Destination', 'Origin'), loc='best')
    plt.show()


def trajectory2(point1, point2, matrix, poly1, poly2, poly3, poly4, poly5):
    cmap = colors.ListedColormap(['white', 'red', 'black'])
    data = np.zeros(matrix)
    for i in range(len(point1)):
        ax = plt.gca()
        data[np.unravel_index(point1[i], matrix)] = 1
        data[poly1[0]:poly1[1] + 1, poly1[2]:poly1[3] + 1] = 2
        data[poly2[0]:poly2[1] + 1, poly2[2]:poly2[3] + 1] = 2
        data[poly3[0]:poly3[1] + 1, poly3[2]:poly3[3] + 1] = 2
        data[poly4[0]:poly4[1] + 1, poly4[2]:poly4[3] + 1] = 2
        data[poly5[0]:poly5[1] + 1, poly5[2]:poly5[3] + 1] = 2

        ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=0.3)
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.6)
        major_ticks = np.arange(-.5, matrix[0], 2)
        minor_ticks = np.arange(-.5, matrix[1], 2)

        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True) 
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.imshow(data, cmap=cmap)

    pp1 = np.unravel_index(point1, matrix)
    x1 = pp1[1]
    y1 = pp1[0]

    pp2 = np.unravel_index(point2, matrix)
    x2 = pp2[1]
    y2 = pp2[0]

    # plot path for combined
    plt.plot(x1, y1, '--', linewidth=1.2, color='k') #k is black
    plt.plot(x2, y2, linewidth=1.2, color='r') # r is red
    plt.plot([40], [0], 'P', color='blue')
    plt.plot([0], [45], 'ob', color='blue')
    plt.legend(('Trajectory Without Bound', 'Trajectory With Bound ', 'Destination', 'Origin'), loc='best')
    plt.show()

    """
The Draw class in the draw.py file provides methods to draw rectangles and circles on a plot. The drawrect method takes an axis ax and a list of coordinates rect and plots the corresponding lines on the plot. The drawcir method takes an axis ax, a tuple of coordinates cir, and a radius r and plots a circle on the plot with the specified center and radius.

The findaction function takes two arguments: point1, a single integer representing a point in the map, and matrix, a 2D array representing the map. It returns a list of actions taken to go from the starting point to the final point. The drawbound function takes four arguments: x1 and y1, lists of x and y coordinates respectively, action, a list of actions taken to go from the starting point to the final point, and ax, the axis on which to plot the actions. It plots the actions taken on the plot using the drawcir and drawrect methods of the Draw class. The delta variable is not used in this file.


    """
