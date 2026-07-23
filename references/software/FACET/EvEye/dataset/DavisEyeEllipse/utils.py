import cv2
import numpy as np


def cal_ellipse_area(major_axis, minor_axis):
    area = np.pi * major_axis * minor_axis / 4

    return area


def convert_to_ellipse(label):
    center = (float(label[1]), float(label[2]))
    axes = (float(label[3]), float(label[4]))
    angle = float(label[5])
    ellipse = (center, axes, angle)
    return ellipse


def get_input_size(keep_resolution, height, width):
    if keep_resolution:
        input_height = (height | 31) + 1
        input_width = (width | 31) + 1
        scale = np.array([input_width, input_height], dtype=np.float32)
    else:
        scale = max(height, width) * 1.0
        input_height, input_width = 256, 256
    return input_height, input_width, scale


def get_dir(src_point, rot_rad):  # Define the direction helper
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)  # Compute the sine and cosine of the rotation angle
    src_result = [0, 0]  # Initialize the result
    src_result[0] = src_point[0] * cs - src_point[1] * sn  # Compute the x coordinate
    src_result[1] = src_point[0] * sn + src_point[1] * cs  # Compute the y coordinate
    return src_result  # Return the computed result


def get_3rd_point(a, b):  # Define the third-point helper
    direct = a - b  # Compute the direction vector
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)  # Return the third point coordinates


def get_affine_transform(  # Define the affine transform matrix helper
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(
        scale, list
    ):  # Check the scale type
        scale = np.array([scale, scale], dtype=np.float32)  # Convert to an array

    scale_tmp = scale  # Temporarily store scale
    src_w = scale_tmp[0]  # Source width
    dst_w = output_size[0]  # Target width
    dst_h = output_size[1]  # Target height

    rot_rad = np.pi * rot / 180  # Convert degrees to radians
    src_dir = get_dir([0, src_w * -0.5], rot_rad)  # Get the source direction
    dst_dir = np.array([0, dst_w * -0.5], np.float32)  # Get the destination direction

    src = np.zeros((3, 2), dtype=np.float32)  # Initialize the source point array
    dst = np.zeros((3, 2), dtype=np.float32)  # Initialize the destination point array
    src[0, :] = center + scale_tmp * shift  # Compute the source point
    src[1, :] = center + src_dir + scale_tmp * shift  # Compute the source point direction
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]  # Compute the destination point
    dst[1, :] = (
        np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    )  # Compute the destination point direction

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])  # Compute the third source point
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])  # Compute the third destination point

    if inv:  # If the inverse transform is required
        trans = cv2.getAffineTransform(
            np.float32(dst), np.float32(src)
        )  # Get the inverse affine transform matrix
    else:
        trans = cv2.getAffineTransform(
            np.float32(src), np.float32(dst)
        )  # Get the affine transform matrix
    return trans  # Return the transform matrix


def gaussian2D(shape, sigma=1):  # Define the 2D Gaussian generator
    m, n = [(ss - 1.0) / 2.0 for ss in shape]  # Compute the center point
    y, x = np.ogrid[-m : m + 1, -n : n + 1]  # Generate grid coordinates
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))  # Compute Gaussian values
    h[h < np.finfo(h.dtype).eps * h.max()] = 0  # Set Gaussian values below the threshold to 0
    return h  # Return the Gaussian distribution


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def affine_transform(pt, t, angle=0, mode='xy'):  # Define the affine transform function
    if mode == 'xy':  # If mode is 'xy'
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T  # Convert the point to homogeneous coordinates
        new_pt = np.dot(t, new_pt)  # Apply the transform matrix
    elif mode == 'ab':  # If mode is 'ab'
        angle = np.deg2rad(angle)  # Convert degrees to radians
        cosA = np.abs(np.cos(angle))  # Compute the cosine value
        sinA = np.abs(np.sin(angle))  # Compute the sine value
        a_x = pt[0] * cosA  # Compute the x coordinate of point a
        a_y = pt[0] * sinA  # Compute the y coordinate of point a
        b_x = pt[1] * sinA  # Compute the x coordinate of point b
        b_y = pt[1] * cosA  # Compute the y coordinate of point b
        new_pt_a = np.array([a_x, a_y, 0.0], dtype=np.float32).T  # Homogeneous coordinates of point a
        new_pt_a = np.dot(t, new_pt_a)  # Apply the transform matrix
        new_pt_b = np.array([b_x, b_y, 0.0], dtype=np.float32).T  # Homogeneous coordinates of point b
        new_pt_b = np.dot(t, new_pt_b)  # Apply the transform matrix
        new_pt = np.zeros((2, 1), dtype=np.float32)  # Initialize the new point
        new_pt[0] = np.sqrt(new_pt_a[0] ** 2 + new_pt_a[1] ** 2)  # Compute the x coordinate of the new point
        new_pt[1] = np.sqrt(new_pt_b[0] ** 2 + new_pt_b[1] ** 2)  # Compute the y coordinate of the new point
    return new_pt[:2]  # Return the first two coordinates of the new point


def gaussian_radius(det_size, min_overlap=0.7):  # Define the Gaussian radius helper
    height, width = det_size  # Get the detection box height and width
    a1 = 1  # Linear-equation coefficient
    b1 = height + width  # Linear-equation constant term
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)  # Linear-equation constant term
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)  # Compute the square root
    r1 = (b1 + sq1) / 2  # Compute the radius
    a2 = 4  # Quadratic-equation coefficient
    b2 = 2 * (height + width)  # Quadratic-equation constant term
    c2 = (1 - min_overlap) * width * height  # Quadratic-equation constant term
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)  # Compute the square root
    r2 = (b2 + sq2) / 2  # Compute the radius
    a3 = 4 * min_overlap  # Cubic-equation coefficient
    b3 = -2 * min_overlap * (height + width)  # Cubic-equation constant term
    c3 = (min_overlap - 1) * width * height  # Cubic-equation constant term
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)  # Compute the square root
    r3 = (b3 + sq3) / 2  # Compute the radius
    return min(r1, r2, r3)  # Return the smallest radius
