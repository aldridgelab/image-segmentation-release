import numpy as np
from scipy.interpolate import splprep, splev, RectBivariateSpline
import warnings


def spline_approximation(n2array, n=200, smooth_factor=1, closed=True):
    warnings.filterwarnings("ignore")
    """
    Updated: 2023.04.12
    Compute a spline approximation of a 2D curve.

    :param n2array: a 2D numpy array of shape (n_points, 2) representing the curve to be approximated
    :param n: number of points of the output polyline/polygon
    :param smooth_factor: smoothing factor for the spline
    :param closed: whether the curve is closed or not
    :return: a 2D numpy array of shape (n, 2) representing the spline approximation of the curve
    """
    # Need at least 4 points for cubic spline (k=3, so m > k requires m >= 4)
    if len(n2array) < 4:
        return n2array

    if closed:
        tck, u = splprep(n2array.T, u=None, s=smooth_factor, per=1)
    else:
        tck, u = splprep(n2array.T, u=None, s=smooth_factor)
    u_new = np.linspace(u.min(), u.max(), n)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.array([x_new, y_new]).T


def bilinear_interpolate_numpy(im, x, y):
    """
    bilinear interpolation 2D
    :param im: target image
    :param x: x coordinates
    :param y: y coordinates
    :return: interpolated data
    """
    h,l = im.shape
    padded = np.zeros((h+1,l+1))
    padded[:h,:l] += im
    im = padded
    x0 = x.astype(int)
    x1 = x0 + 1
    y0 = y.astype(int)
    y1 = y0 + 1
    Ia = im[x0,y0]
    Ib = im[x0,y1]
    Ic = im[x1,y0]
    Id = im[x1,y1]
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    return (Ia*wa) + (Ib*wb) + (Ic*wc) + (Id*wd)


def unit_perpendicular_vector(data, closed=True):
    """
    Compute the unit perpendicular vector of a set of 2D points.

    :param data: A numpy array of shape (n, 2) representing the 2D points.
    :param closed: A boolean indicating whether the curve is closed or not.
    :return: A numpy array of shape (n, 2) representing the unit perpendicular vector of the curve.
    """
    # Need at least 2 points to compute perpendicular vectors
    if len(data) < 2:
        return np.array([])
    p1 = data[1:]
    p2 = data[:-1]
    dxy = p1 - p2
    ang = np.arctan2(dxy.T[1], dxy.T[0]) + 0.5 * np.pi
    dx, dy = np.cos(ang), np.sin(ang)
    unit_dxy = np.array([dx, dy]).T
    if len(unit_dxy) == 0:
        return np.array([])
    if not closed:
        unit_dxy = np.concatenate([[unit_dxy[0]], unit_dxy])
    else:
        unit_dxy = np.concatenate([unit_dxy, [unit_dxy[0]]])
    return unit_dxy


def intersect_lines(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Compute the intersection point between two lines.

    :param x1: A float representing the x-coordinate of the first point of the first line.
    :param y1: A float representing the y-coordinate of the first point of the first line.
    :param x2: A float representing the x-coordinate of the second point of the first line.
    :param y2: A float representing the y-coordinate of the second point of the first line.
    :param x3: A float representing the x-coordinate of the first point of the second line.
    :param y3: A float representing the y-coordinate of the first point of the second line.
    :param x4: A float representing the x-coordinate of the second point of the second line.
    :param y4: A float representing the y-coordinate of the second point of the second line.
    :return: A tuple of floats representing the x and y coordinates of the intersection point.
    """
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3
    intersect_x = (B1 * C2 - B2 * C1) / (A2 * B1 - A1 * B2)
    intersect_y = (A1 * C2 - A2 * C1) / (B2 * A1 - B1 * A2)
    return intersect_x, intersect_y


def line_contour_intersection(p1, p2, contour):
    """
    Compute the intersection point(s) between a line and a contour.

    :param p1: A tuple of floats representing the (x, y) coordinates of the start point of the line.
    :param p2: A tuple of floats representing the (x, y) coordinates of the end point of the line.
    :param contour: A numpy array of shape (n, 2) representing the points of the contour.
    :return: A numpy array of shape (m, 2) representing the x and y coordinates of the intersection point(s).
    """
    v1, v2 = contour[:-1], contour[1:]
    x1, y1 = v1.T
    x2, y2 = v2.T
    x3, y3 = p1
    x4, y4 = p2
    xy = np.array(intersect_lines(x1, y1, x2, y2, x3, y3, x4, y4)).T
    dxy_v1 = xy - v1
    dxy_v2 = xy - v2
    dxy = dxy_v1 * dxy_v2
    intersection_points = xy[np.where(np.logical_and(dxy[:, 0] < 0, dxy[:, 1] < 0))]
    if len(intersection_points) > 2:
        dist = np.sum(np.square(np.tile(p1, (len(intersection_points), 1)) - intersection_points),
                      axis=1)
        intersection_points = intersection_points[np.argsort(dist)[0:2]]
    return intersection_points

def distance(v1, v2):
    """
    Euclidean distance of two points
    :param v1: vecorized coordinate of the first point.
    :param v2: vecorized coordinate of the second point.
    :return: distance between two points.
    """
    return np.sqrt(np.sum((np.array(v1) - np.array(v2)) ** 2))


def distance_matrix(data1, data2):
    """
    Compute the Euclidean distance matrix between two sets of 2D points.

    :param data1: A numpy array of shape (n, 2) representing the first set of 2D points.
    :param data2: A numpy array of shape (m, 2) representing the second set of 2D points.
    :return: A numpy array of shape (n, m) representing the Euclidean distance matrix between the two sets of points.
    """
    x1, y1 = data1.T
    x2, y2 = data2.T
    dx = x1[:, np.newaxis] - x2
    dy = y1[:, np.newaxis] - y2
    dxy = np.sqrt(dx ** 2 + dy ** 2)
    return dxy


def point_line_orientation(l1,l2,p):
    """
    Compute the orientation of a point with respect to a line.

    :param l1: A tuple of floats representing the (x, y) coordinates of the first point of the line.
    :param l2: A tuple of floats representing the (x, y) coordinates of the second point of the line.
    :param p: A tuple of floats representing the (x, y) coordinates of the point.
    :return: A float representing the orientation of the point with respect to the line.
    """
    return np.sign(((l2[0] - l1[0]) * (p[1] - l1[1]) - (l2[1] - l1[1]) * (p[0] - l1[0])))


def interpolate_2Dmesh(data_array, smooth=1):
    """
    Interpolate a 2D mesh of data using a rectangular bivariate spline.

    :param data_array: A numpy array of shape (n, m) representing the 2D mesh of data to interpolate.
    :param smooth: A float representing the smoothing factor to use for the spline interpolation.
    :return: A RectBivariateSpline object representing the interpolated mesh of data.
    """
    x_mesh = np.linspace(0, data_array.shape[0] - 1, data_array.shape[0]).astype(int)
    y_mesh = np.linspace(0, data_array.shape[1] - 1, data_array.shape[1]).astype(int)
    return RectBivariateSpline(x_mesh, y_mesh, data_array, s=smooth)


def bend_angle_closed(data, window=3, deg=True):
    """
    Compute the bending angle of a closed 2D curve.

    :param data: A numpy array of shape (n, 2) representing the 2D curve.
    :param window: An integer representing the size of the window to use for computing the bending angle.
    :param deg: A boolean indicating whether to return the angle in degrees (True) or radians (False).
    :return: A float representing the bending angle of the curve.
    """
    p1 = np.concatenate((data[-window:],data[:-window])).T
    p2 = data.copy().T
    p3 = np.concatenate((data[window:],data[0:window])).T
    p1p2 = p1[0]*1+p1[1]*1j - (p2[0]*1+p2[1]*1j)
    p1p3 = p1[0]*1+p1[1]*1j - (p3[0]*1+p3[1]*1j)
    return np.angle(p1p3/p1p2, deg=deg)


def bend_angle_open(data, window=3, deg=True):
    """
    Compute the bending angle of an open 2D curve.

    :param data: A numpy array of shape (n, 2) representing the 2D curve.
    :param window: An integer representing the size of the window to use for computing the bending angle.
    :param deg: A boolean indicating whether to return the angle in degrees (True) or radians (False).
    :return: A float representing the bending angle of the curve.
    """
    p1 = data[:-2*window].T
    p2 = data[window:-window].T
    p3 = data[2*window:].T
    p1p2 = p1[0]*1+p1[1]*1j - (p2[0]*1+p2[1]*1j)
    p2p3 = p2[0]*1+p2[1]*1j - (p3[0]*1+p3[1]*1j)
    return np.angle(p2p3/p1p2, deg=deg)


def angle_between_vectors(v1, v2):
    """
    Compute the angle between two vectors.

    :param v1: A numpy array representing the first vector.
    :param v2: A numpy array representing the second vector.
    :return: A float representing the angle between the two vectors in radians.
    """
    d1 = np.sqrt(np.sum(v1 ** 2))
    d2 = np.sqrt(np.sum(v2 ** 2))
    return np.arccos(np.dot(v1, v2) / (d1 * d2))


def measure_length(data, pixel_microns=1):
    """
    Compute the length of a 2D polyline.

    :param data: A numpy array of shape (n, 2) representing the 2D polyline.
    :param pixel_microns: A float representing the conversion factor from pixels to microns.
    :return: A float representing the length of the polyline in microns.
    """
    v1,v2 = data[:-1], data[1:]
    length = np.sqrt(np.sum((np.array(v1) - np.array(v2)) ** 2, axis=1)).sum()*pixel_microns
    return(length)


def line_contour_intersect_matrix(line, contour, orthogonal_vectors=None):

    """
    Compute the intersection point(s) between a line and a contour.
    Previously named "intersect_matrix"

    :param line: A numpy array of shape (2, 2) representing the start and end points of a straight line.
    :param contour: A numpy array of shape (n, 2) representing the points of the contour.
    :param orthogonal_vectors: A numpy array of shape (n, 2) representing the unit perpendicular vectors of the contour.
    :return: A tuple of numpy arrays representing the x and y coordinates of the intersection point(s).
    """
    # Need at least 2 points in line and contour
    if len(line) < 2 or len(contour) < 2:
        return np.array([]), np.array([])

    if orthogonal_vectors is None:
        dxy = unit_perpendicular_vector(line, closed=False)
    else:
        dxy = orthogonal_vectors
    # Check if dxy is valid
    if len(dxy) == 0:
        return np.array([]), np.array([])
    v1, v2 = contour[:-1], contour[1:]
    x1, y1 = v1.T
    x2, y2 = v2.T
    x3, y3 = line.T
    perp_xy = line + dxy
    x4, y4 = perp_xy.T
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    A1B2 = A1[:, np.newaxis] * B2
    A1C2 = A1[:, np.newaxis] * C2
    B1A2 = B1[:, np.newaxis] * A2
    B1C2 = B1[:, np.newaxis] * C2
    C1A2 = C1[:, np.newaxis] * A2
    C1B2 = C1[:, np.newaxis] * B2

    intersect_x = (B1C2 - C1B2) / (B1A2 - A1B2)
    intersect_y = (A1C2 - C1A2) / (A1B2 - B1A2)
    return intersect_x, intersect_y

