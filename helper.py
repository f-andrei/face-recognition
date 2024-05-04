import numpy as np


def shape_to_landmark_list(face_landmarks):
    landmark_list = []
    for i in range(0, 5):
        landmark_list.append((i, (face_landmarks.part(i).x, face_landmarks.part(i).y)))
    return landmark_list

def get_eyes_nose(shape_landmarks):
    nose = shape_landmarks[4][1]
    l_eye = ((shape_landmarks[3][1][0] + shape_landmarks[2][1][0]) // 2,
                (shape_landmarks[3][1][1] + shape_landmarks[2][1][1]) // 2)
    r_eye = ((shape_landmarks[1][1][0] + shape_landmarks[0][1][0]) // 2,
                 (shape_landmarks[1][1][1] + shape_landmarks[0][1][1]) // 2)
    return nose, l_eye, r_eye

def eucliean_distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def cosine_formula(line1, line2, line3):
    cos_angle = -(line3 ** 2 - line2 ** 2 - line1 ** 2) / (2 * line2 * line1)
    return cos_angle

def rotate_point(origin, point, angle):
    origin_x, origin_y = origin
    point_x, point_y = point

    rotated_x = origin_x + np.cos(angle) * (point_x - origin_x) - np.sin(angle) * (point_y - origin_y)
    rotated_y = origin_y + np.sin(angle) * (point_x - origin_x) + np.cos(angle) * (point_y - origin_y)
    return rotated_x, rotated_y

def is_between(point1, point2, point3, extra_point):
    c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (extra_point[0] - point1[0])
    c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (extra_point[0] - point2[0])
    c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (extra_point[0] - point3[0])
    if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
        return True
    else:
        return False