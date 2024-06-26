"""Module to generate .obj file from bronchus coords outer shell

Needs stage-02 (reduced_model.npz) as input.

This file expects the first argument to be a path to the input folder (not including the
reduced_model.npz). The second argument needs to be a path to the output file.

It does so by going through every bronchus point in the reduced_model.npz and checking
each of the 6 neighboring points whether it is empty. If it is indeed empty then
it adds the points which haven't been added yet on that face and also adds the face.

Then it saves everything into a .obj file with the format of [patient_it].obj. This file can
be imported into blender, 3D printed, visualized and many other nice things.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Union

import numpy as np

from spine_segmentation.visualisation.color import Color

# from spine.spine_types.arraytype import Vol_3D_I, Mat_4x4_F, List_Vec_3D_F, Vec_3D_F
# from spine.spine_types.color import Color
# from spine.util.geom_types import Ellipsoid


def generate_obj(
    output_data_path: Path,
    model,
    accepted_types: Set[int] = {},
    color_mask=None,
    type_to_color: Union[Color, Dict[int, Color]] = {},
    rot_mat=None,
    num_decimal_digits: int = 4,
):
    """Saves a .obj obj_file given the model, the accepted types and a name

    output_data_path is a pathlib Path, this is the full path the obj_file will be saved as

    model is the 3D numpy array model of the lung or whatever object you
    want to convert to 3D

    accepted_types is a list or set which types should be looked for, other
    types will be ignored. If empty set then everything except for 0 will be
    accepted

    color_to_rgb_tuple is a Dict which maps the color id used in color mask to a tuple
    of rgb values. This color will be used to color the object with that color.
    (e.g. {1: (255, 120, 150)})

    color_mask is a model with the same shape as model, but its numbers represent
    groups of colors/materials which will be added by this script

    rot_mat is a rotation matrix. Each point p=(x,y,z) is rotated by rot_mat @ p or left unchanged if None
    """
    # color_to_rgb_tuple = {}
    # for color_id, color in type_to_color.items():
    #     color_to_rgb_tuple[color_id] = color.floats()[:3]

    output_data_path = Path(output_data_path)

    # Add padding
    model = np.pad(model, 1, mode="constant", constant_values=0)
    if color_mask is not None:
        color_mask = np.pad(color_mask, 1, mode="constant", constant_values=0)

    def color_from_type(type_):
        if isinstance(type_to_color, Color):
            col = type_to_color
        else:
            col = type_to_color.get(material, Color.random())
        return col.floats()[:3]

    vertices = {}
    faces: Dict[int, List[List[int]]] = defaultdict(list)

    reference_shape = model.shape
    model = model.copy()
    if accepted_types:
        for remove in set(np.unique(model)) - accepted_types - {0}:
            model[model == remove] = 0

    index = 1
    # Iterate over each axis and pos/neg directions, then roll the model over, afterwards subtracting these.
    # This causes there to be only -1 and 1 values where there is air, meaning there a face should be added.
    # Though we only look at the 1 values, since these are actually in the model (the others are outside, which means
    # their color map will be wrong)
    for axis in range(3):
        for pos_or_neg in [-1, 1]:
            diff = np.roll(model, -pos_or_neg, axis=axis)
            model_diff = np.where(model > diff)
            coords = []
            for d1, d2 in [(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]:
                coords.append(list(map(lambda t: t.astype(float), np.copy(model_diff))))
                coords[-1][axis] += pos_or_neg / 2
                coords[-1][(axis + 1) % 3] += d1
                coords[-1][(axis + 2) % 3] += d2

            # Example shape after transpose: (11523, 4, 3) - face_coords is an array of faces,
            # each face is a list of 4 points with 3 coordinates.
            faces_coords = np.transpose(coords, axes=(2, 0, 1))
            vertex_coords = np.transpose(model_diff)
            for vertex_coord, face_coords in zip(vertex_coords, faces_coords):
                curr_face = []
                for face_coord in map(tuple, face_coords):
                    if face_coord not in vertices:
                        vertices[face_coord] = index
                        index += 1
                    curr_face.append(vertices[face_coord])
                material = color_mask[tuple(vertex_coord)] if color_mask is not None else 0
                faces[material].append(curr_face)

    # make to numpy for easier usage later
    vertices = np.array([np.array(v) for v in vertices])
    vertices = normalize(vertices, rot_mat=rot_mat)

    # Write vertices and faces to obj_file
    material_path = output_data_path.with_suffix(".mtl")
    with open(material_path, "w") as mat_file:
        for material in faces:
            mat_file.write(f"newmtl mat{material}\n")
            mat_file.write("Ns 96.078431\n")
            mat_file.write("Ka 1.000000 1.000000 1.000000\n")
            rgb = color_from_type(material)
            mat_file.write(f"Kd {' '.join(map(str, rgb))}\n")
            mat_file.write("Ks 0.500000 0.500000 0.500000\n")
            mat_file.write("Ke 0.000000 0.000000 0.000000\n")
            mat_file.write("Ni 1.000000\n")
            mat_file.write("d 1.000000\n")
            mat_file.write("illum 2\n\n")
    with open(output_data_path, "w") as obj_file:
        obj_file.write("# .obj generated by Airway")
        obj_file.write(f"mtllib {material_path.name}\n")
        obj_file.write("# Vertices\n")
        # original was [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
        vertexformat = f"v {{:.{num_decimal_digits}f}} {{:.{num_decimal_digits}f}} {{:.{num_decimal_digits}f}}\n"
        for x, y, z in vertices:
            obj_file.write(vertexformat.format(x, y, z))

        obj_file.write("\n# Faces\n")
        for material, faces_with_material in faces.items():
            obj_file.write(f"usemtl mat{material}\n")
            for a, b, c, d in faces_with_material:
                obj_file.write(f"f {a} {b} {c} {d}\n")


def normalize(vertices, reference_shape: np.ndarray = (18, 320, 896), rot_mat=None):
    if reference_shape is not None:
        # Shift to middle of the space
        vertices -= np.array(reference_shape) / 2
        # Scale to [-10..10]
        vertices *= 20 / np.max(reference_shape)
    # If available: transform
    # Note: since this is applied afterwards, points can be out of [-10..10]
    if rot_mat is not None:
        # Add 4th dimension as 1 for transformation matrix
        ones_vec3_to_vec4 = np.ones((len(vertices), 1))
        vertices = np.append(vertices, ones_vec3_to_vec4, axis=1)
        # Transform
        vertices = vertices @ np.transpose(rot_mat)
        # Remove 4th dimension from vectors
        vertices = vertices[:, :-1]
    return vertices


# def generate_ellipsoid_obj(
#         filepath: Path,
#         ellipsoid: Ellipsoid,
#         reference_shape: Vec_3D_F,
#         rot_mat: Mat_4x4_F = None,
#         n_latitude: int = 10,
#         n_longitude: int = 10,
# ):
#     with open(filepath, "w") as file:
#         center, radii = ellipsoid.center, ellipsoid.radii
#         vertices = [center + radii * [0, 0, 1], center - radii * [0, 0, 1]]  # Top and bottom vertices
#         curr_vertex_index = 3  # 1 and 2 are the above points (.obj files are 1-indexed)
#         latitude_point_indices = []  # Fill with lists of indexes for each latitude, used to connect faces later
#
#         for lon_angle in np.linspace(0, 2 * np.pi, n_longitude, endpoint=False):  # Traverse angle along longitude
#             latitude_point_indices.append(list(range(curr_vertex_index, curr_vertex_index + n_latitude - 2)))
#             curr_vertex_index += n_latitude - 2
#             for lat_angle in np.linspace(0, np.pi, n_latitude)[1:-1]:  # Skip existing first and last points
#                 polar_angle = np.array(
#                     [np.sin(lat_angle) * np.cos(lon_angle), np.sin(lat_angle) * np.sin(lon_angle), np.cos(lat_angle)]
#                 )
#                 vertices.append(center + radii * polar_angle)
#         # Normalize twice, once with inverse, then translate with reference shape, then retransform to align coordinates
#         for vertex in normalize(normalize(vertices, None, np.linalg.inv(rot_mat)), reference_shape, rot_mat):
#             file.write(f"v {' '.join(map(str, vertex))}\n")
#         for left, right in zip(latitude_point_indices, latitude_point_indices[1:] + latitude_point_indices[:1]):
#             file.write(f"f 1 {left[0]} {right[0]}\n")  # First triangle to top point
#             file.write(f"f 2 {left[-1]} {right[-1]}\n")  # Last triangle to bottom point
#             for left2points, right2points in zip(zip(left, left[1:]), zip(right, right[1:])):
#                 file.write(f"f {' '.join(map(str, left2points + right2points[::-1]))}\n")  # Connect squares
