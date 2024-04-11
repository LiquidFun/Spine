import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Literal, Tuple, Union

import numpy as np

from spine_segmentation.spine_types.line_segment import LineSegment
from spine_segmentation.spine_types.plane import Plane
from spine_segmentation.visualisation.blender.gen_obj import generate_obj, normalize
from spine_segmentation.visualisation.color import Color
from spine_segmentation.visualisation.color_palettes import get_alternating_palette

_geom_scale = (3.3, 0.8, 0.8)
_geom_transform_mat = np.array(
    [
        [0, _geom_scale[1], 0, 0],
        [0, 0, _geom_scale[2], 0],
        [_geom_scale[0], 0, 0, 0],
        [0, 0, 0, 1],
    ]
)


class BlenderCollection:
    def __init__(self, name: str, objects: List = [], *, hidden: bool = False):
        self.name = name
        self.hidden = hidden
        self.objects = objects

    def script_args(self):
        hidden_str = "-hidden" if self.hidden else ""
        return [f"--collection{hidden_str}", self.name]


def create_3d_models_and_labels(
    spine, *, directory: Union[str, Path], color_mask
) -> Tuple[List[Path], Dict[str, Tuple]]:
    """Creates the 3D .obj files for the spine itself, filters rois, such that only ones in the mask are included"""
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    paths = []

    # def mask_overlaps_with(roi):
    #     """Return True if there is any overlap between the mask and the given roi"""
    #     roi_bounds = [self._index3D_to_image2D(p) for p in roi.boundingbox_raw]
    #     roi_bounds_min, roi_bounds_max = np.min(roi_bounds, axis=0), np.max(roi_bounds, axis=0)
    #     mask_bounds_min, mask_bounds_max = self._mask_cropping_box[:2], self._mask_cropping_box[2:]
    #     bounds_min = np.max([mask_bounds_min, roi_bounds_min], axis=0)
    #     bounds_max = np.min([mask_bounds_max, roi_bounds_max], axis=0)
    #     return all((bounds_max - bounds_min) > 0)

    # filtered_vertebrae = list(filter(mask_overlaps_with, self._spine.vertebras))
    # filtered_discs = list(filter(mask_overlaps_with, self._spine.discs))

    name_and_id_set_pairs = [
        ("Vertebrae", spine, set(range(1, 100, 2))),
        ("Discs", spine, set(range(2, 100, 2))),
        # ("Spinal_canal", {self.spine.idx_spinal_canal}, (self._spine.seg_canal != 0) * self.spine.idx_spinal_canal),
    ]
    for name, seg_3d, id_set in name_and_id_set_pairs:
        obj_path = directory / f"{name}.obj"
        print(f"Generating {obj_path} {seg_3d.shape}")
        # seg_3d = seg_3d[::2, ::2, ::2]
        generate_obj(
            obj_path,
            seg_3d,
            id_set,
            seg_3d,
            type_to_color=color_mask,
            rot_mat=_geom_transform_mat,
        )
        paths.append(obj_path)

    return paths


def _write_obj_if_possible_for_object(obj: object, filepath: Path) -> bool:
    """Create a .obj file for the given object if possible, return True if yes"""

    def write_obj_file(points, face_or_line: Literal["f", "l"], start_at=1, rot_mat=_geom_transform_mat):
        points = normalize(points, rot_mat=rot_mat)
        with open(filepath, "a") as obj_file:
            for point in points:
                obj_file.write("v " + " ".join(map(str, point)) + "\n")
            obj_file.write(f"{face_or_line} " + " ".join(map(str, range(start_at, len(points) + start_at))) + "\n")

    if isinstance(obj, LineSegment):  # or isinstance(obj, Polygon):
        write_obj_file(obj.points, "l")
    if isinstance(obj, Plane):
        write_obj_file(obj.points, "f")
    if isinstance(obj, np.ndarray):
        if len(obj.shape) == 3:
            indices_for_3D = set(np.unique(obj)) - {0}  # Do not use empty points for constructing the object
            generate_obj(filepath, obj, indices_for_3D, obj, rot_mat=_geom_transform_mat)
    # if isinstance(obj, Ellipsoid):
    #     generate_ellipsoid_obj(filepath, obj, self._spine.seg.shape, self._geom_transform_mat)
    # if isinstance(obj, AxisParallelCuboid):
    #     for index, face in enumerate(obj.faces):
    #         write_obj_file(np.array(face), "f", start_at=1 + index * 4)
    return filepath.exists()


def open_in_blender(
    spines: Union[np.ndarray, Dict[str, np.ndarray]],
    collections: List[BlenderCollection] = [],
    *,
    smooth_spine: bool = True,
    export_to_instead: Union[bool, str, Path] = False,
    render_to_instead: Union[bool, str, Path] = False,
):
    """Opens a 3D view in blender of all added objects. Useful for debugging. Blender 2.80+ is required.
    Pauses program until the blender process finishes.

    @param spines: the spine to show, 3D numpy array
    @param collections: list of blender collections which will be rendered additionally
    @param smooth_spine: whether a smoothing modifier should be applied to spine, otherwise raw voxels are shown
    @param export_to_instead: instead of opening Blender, export a .dae model to the given path. If True simply
           export to the temp directory with the name "model.dae".
    @param render_to_instead: instead of opening Blender, render the scene to the given path. If True simply
           render to the temp directory with the name "render.png".
    @return: None
    """

    alternating_palette = get_alternating_palette()

    prefix_to_color_matching = {
        "instances": alternating_palette,
        "output": {1: Color("orange"), 2: Color("lightgreen")},
        "spine": {1: Color("orange"), 2: Color("lightgreen")},
        "gt": {1: Color("red"), 2: Color("black")},
    }

    if not isinstance(spines, dict):
        spines = {"spine": spines}
    with TemporaryDirectory(prefix="spine_blender_") as tmp_dir:
        script_path = Path(__file__).absolute().parent / "blender_script.py"
        blender_args = ["blender", "-P", str(script_path)]
        if export_to_instead or render_to_instead:
            blender_args.extend(["--background"])

        script_args = []
        for prefix, spine in spines.items():
            script_args.extend(["--collection", prefix.capitalize()])
            color_mask = prefix_to_color_matching.get(prefix, None)
            obj_paths = create_3d_models_and_labels(spine, directory=Path(tmp_dir) / prefix, color_mask=color_mask)
            for obj_path in obj_paths:
                spine_arg = "--object-smooth" if smooth_spine else "--object"
                script_args.extend([spine_arg, str(obj_path)])

        if render_to_instead:
            path = Path(tmp_dir) / "render.png" if render_to_instead is True else Path(render_to_instead)
            path.parent.mkdir(parents=True, exist_ok=True)
            script_args.extend(["--render-to", str(path)])

        for collection in collections:
            script_args.extend(collection.script_args())
            collection_dir = Path(tmp_dir) / collection.name
            collection_dir.mkdir(parents=True)

            for index, object_3D in enumerate(collection.objects, 1):
                filename = getattr(object_3D, "name", f"{collection.name[:-1]}{index}")
                object_3D_path = collection_dir / f"{filename}.obj"
                print(collection, object_3D_path)
                if _write_obj_if_possible_for_object(object_3D, object_3D_path):
                    script_args.extend(["--object", str(object_3D_path)])

        if export_to_instead:
            path = Path(tmp_dir) / "model.dae" if render_to_instead is True else Path(render_to_instead)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Export the blender composed model
            script_args.extend(["--export-to", str(path)])
        #     # Create the json file with the center names
        #     vert_centers_path = export_to_instead.with_name("vertebrae_centers.json")
        #     if not vert_centers_path.exists():
        #         with open(vert_centers_path, "w") as file:
        #             import json
        #             json.dump(center_labels, file)

        arg_separator = ["--"]
        print(blender_args + arg_separator + script_args)
        subprocess.run(blender_args + arg_separator + script_args)
