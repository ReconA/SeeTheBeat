import math
import random
import numpy as np
import cv2
import copy


def points_in_circle(x0, y0, r, n=100):
    """ Get a list of n points that are on the circle arc.
    :param x0: x-coordinate of circle center
    :param y0: y-coordinate of circle center
    :param r: Circle radius
    :param n: Number of points to generate.
    :return: A list of points on circle arc.
    """
    return list([(math.cos(2*math.pi/n*x)*r+ x0, math.sin(2*math.pi/n*x)*r + y0) for x in range(0,n+1)])


def random_point_in_circle(x0, y0, r):
    """ Get a random point inside the circle with center (x0,y0) and radius r.
    :param x0: x-coordinate of circle center.
    :param y0: y-coordinate of circle center.
    :param r: Circle radius
    :return: x,y : Random point inside the circle.
    """
    t = random.uniform(0, 2*math.pi)
    c = random.uniform(-1,1)
    x = x0 + c * (x0 + r*math.cos(t))
    y = y0 + c * (y0 + r*math.sin(t))

    return x,y


def calculate_distance(p1, p2):
    """
    Calculate Euclidean distance between p1 and p2.
    :param p1: Tuple of two floats.
    :param p2: Tuple of two floats.
    :return: Eucliedean distance between p1 and p2.
    """
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def beat(x0, y0, r, circle_points):
    """

    :param x0:
    :param y0:
    :param r:
    :param circle_points:
    :return:
    """
    origin = random_point_in_circle(x0, y0, r)
    new_points = list([push(p, origin) for p in circle_points])

    # If the distance between two consecutive points exceeds the threshold, create a new point between them.
    distance_threshold = 5
    i = 0
    while i < len(new_points):
        if calculate_distance(new_points[i], new_points[i-1]) > distance_threshold:
            x = (new_points[i-1][0] + new_points[i][0])/2
            y = (new_points[i-1][1] + new_points[i][1])/2
            new_points.insert(i, (x,y))
        i += 1
    return new_points


def calculate_force(distance):
    return 100/(distance**2)


def push(p, origin):
    distance = calculate_distance(p, origin)
    force = calculate_force(distance)
    new_position = (p[0]+(p[0] - origin[0])*force, p[1]+(p[1] - origin[1])*force)
    return new_position


def get_points_inside(canvas, x, y, tempo):

    r = 100
    points = points_in_circle(x, y, r)

    for t in range(int(tempo)):
        points = beat(x, y, r, points)

    x_size, y_size, _ = canvas.shape
    data = np.zeros((x_size, y_size, 3), dtype=np.uint8)

    line_color = (255, 0, 0)
    for i in range(0, len(points)):
        line_start = points[i]
        line_end = points[(i+1)%len(points)]
        cv2.line(data, (int(line_start[0]), int(line_start[1])),(int(line_end[0]), int(line_end[1])), line_color, 1)

    fill_color = (0, 100, 0)

    data, filled_coordinates = flood_fill(data, x, y, fill_color)

    from matplotlib import pyplot as plt
    plt.imshow(data, interpolation='nearest')
    plt.show()

    return filled_coordinates


def flood_fill(data, x, y, fill_value):
    """
    Use flood fill algorithm to get all coordinates inside the created shape.
    :param data: An image that contains a closed shape.
    :param x: x-coordinate of a point inside the shape.
    :param y: y-coordinate of a point inside the shape.
    :param fill_value: What color the algorithm uses to fill the shape. A tuple of 3 integers.
    :return: data: The input image with the a filled shape.
             filled_coordinates: a <int, list> dictionary of all coordinates that were filled.
    """
    filled_coordinates = {}
    print("flood fill from ", x, y)
    x0 = x
    y0 = y
    x_size, y_size,_ = data.shape
    orig_value = copy.deepcopy(data[y, x])

    stack = {(x, y)}
    if np.array_equal(fill_value, orig_value):
        raise ValueError("Filling region with same value "
                         "already present is unsupported. "
                         "Did you already fill this region?")
    while stack:
        x, y = stack.pop()
        if np.array_equal(data[y, x], orig_value):
            data[y, x] = fill_value
            if y not in filled_coordinates.keys():
                filled_coordinates[y] = [x]
            else:
                filled_coordinates[y].append(x)
            if x > 0:
                stack.add((x - 1, y))
            if x < (x_size - 1):
                stack.add((x + 1, y))
            if y > 0:
                stack.add((x, y - 1))
            if y < (y_size - 1):
                stack.add((x, y + 1))

    data[y0,x0] = [255, 255, 255]
    return data, filled_coordinates
