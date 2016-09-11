from swampy.TurtleWorld import *
import math


def polygon(turtle_name, length=50., n=6):
    """Draws a polygon"""
    print turtle_name
    for i in range(n):
        fd(turtle_name, length)
        lt(turtle_name, 360. / n)


def circle(turtle_name, radius=50.):
    num_segments = 30
    polygon(turtle_name, radius * math.pi / num_segments, num_segments)

world = TurtleWorld()
bob = Turtle()
bob.delay = .1
polygon(bob, 50, 6)
sue = Turtle()
sue.delay = .01
circle(sue, 100)

wait_for_user()