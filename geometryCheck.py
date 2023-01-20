import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point
import math

class geometry:
    ###################### check collision between rect and circle #################################################
    def dist(self, p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        px = x2 - x1
        py = y2 - y1

        something = px * px + py * py

        u = ((x3 - x1) * px + (y3 - y1) * py) / (float(something)+1e-8)
        if u > 1:
            u = 1
        elif u < 0:
            u = 0

        x = x1 + u * px
        y = y1 + u * py

        dx = x - x3
        dy = y - y3

        dist = math.sqrt(dx * dx + dy * dy)
        return dist

    def Flagrectc(self, rect, c, r): #check for collisions between the UAV's safety bound and obstacles in the grid
        rect = rect
        c = c
        r = r

        distances = [self.dist(rect[i], rect[j], c) for i, j in zip([0, 1, 2, 3], [1, 2, 3, 0])]
        point = Point(c)
        polygon = Polygon(rect)

        flag = 0
        if any(d < r for d in distances) == True:
            flag = 1
        if any(d < r for d in distances) == False and polygon.contains(point) == True:
            flag = 1  # type: int
        return flag

    ####################### check collision between 2 rect #####################
    def Flag2rect(self, poly1, poly2):
        polygons = [Polygon(poly1), Polygon(poly2)]
        flag = 0
        if polygons[0].intersects(polygons[1]) == True and polygons[0].touches(polygons[1]) == False:
            flag = 1
        return flag

    ####################### check collision between 2 circle ##################
    def Flag2cir(self, c1, r1, c2, r2):
        flag = 0
        if np.linalg.norm(np.subtract(c1, c2)) < r1 + r2:
            flag = 1
        return flag
"""
In the geometryCheck.py file, the geometry class contains several methods for checking for collisions between different shapes. The dist method calculates the distance between a line segment and a point, and the Flagrectc, Flag2rect, and Flag2cir methods check for collisions between a rectangle and a circle, two rectangles, and two circles, respectively.

The UAVEnv class in the environment file uses the geometry class from the geometryCheck.py file to check for collisions between different objects in the environment, such as the safety bound and obstacles. The Draw class from the draw.py file is used to draw the objects in the environment, including the safety bound and obstacles.

In the __init__ method of the UAVEnv class, the geometry class is imported and instantiated as g, and it is used to check for collisions between objects in the environment by calling methods such as Flagrectc, Flag2rect, and Flag2cir. The Draw class is also imported and used to draw the objects in the environment.

In the polyFlag method of the UAVEnv class, the g object is used to check for collisions between the safety bound and obstacles in the environment by calling the Flagrectc method. This method returns a flag value that indicates whether there is a collision between the objects.
"""