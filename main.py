import numpy as np
from shapely import LineString
from shapely import Point
from shapely import line_interpolate_point


def get_unit_vector(a: np.array, b: np.array) -> np.array:
    """get the unit vector point in the direction point A -> point B"""
    vector = np.subtract(b, a)
    unit_v = vector / np.linalg.norm(vector)
    if not 0.99 < np.linalg.norm(unit_v) < 1.01:
        raise ValueError
    return unit_v[0]


def rotate_90_deg(start_point: np.array, end_point: np.array, clockwise: bool) -> np.array:
    """find the coordinates of the end_point
    - rotated 90 degrees around the start point,
    - in the given direction (clockwise or counterclockwise)
    """
    vector = np.subtract(end_point, start_point)

    if clockwise:
        rot = np.array([[0, 1], [-1, 0]])
        return start_point + vector @ rot
    else:
        rot = np.array([[0, -1], [1, 0]])
        return start_point + vector @ rot


class PlineString(LineString):

    def __init__(self, *args, left_clockwise=True, **kwargs):
        super().__init__(*args, **kwargs)
        # if true, clockwise turn from the first point is left
        self.left_clockwise = left_clockwise

    def point_at_distance(self, d: float) -> Point:
        """interpolate a point at a given distance of the starting point of the linestring"""
        point = line_interpolate_point(self, d)
        assert isinstance(Point, point)
        return point

    def get_perpendicular_unit_vector(self, d: float, left=True, delta=10):
        """get a unit vector
         - locally perpendicular to the plinestring at +- delta units
         - at the point at distance d
         - in the given direction (left or right)
         """

        # create a local segment from d to d+10 m to find the perpendicular to
        point = list(self.point_at_distance(d).coords)
        end_point = list(self.point_at_distance(d + delta).coords)

        if left:
            clockwise = self.left_clockwise
        else:
            clockwise = not self.left_clockwise

        # find a point in this direction
        new_end_point = rotate_90_deg(np.array(point), np.array(end_point), clockwise)
        return get_unit_vector(point, new_end_point)
