"""Module which initializes blender scene with objects and parses arguments

This script is meant to be used in conjunction with blender.
It can be added with the -P flag when running blender on
the command line, blender will run this script once the scene has been initialized.
Note that this is might be a different python version, using the global
pip packages instead of the local venv, so ideally no external
packages should be used (such as numpy).
"""

import argparse
import math
import sys
from functools import partial
from pathlib import Path

cameras = []


def load_obj(path):
    """Loads .obj file and returns the object"""
    name = Path(path).name.replace(".obj", "")
    if not Path(path).exists():
        print(f"File {path} does not exist! Skipping it!")
        return None
    bpy.ops.import_scene.obj(filepath=path)
    return bpy.context.selected_objects[-1]


def select(obj, active=False):
    obj.select_set(True)
    if active:
        bpy.context.view_layer.objects.active = obj


def hide(obj, viewport=False, selection=False, render=False):
    obj.hide_viewport = viewport
    obj.hide_render = render
    obj.hide_select = selection


def make_obj_smooth(obj, iterations=5, factor=2):
    """Adds smoothing modifier in Blender"""
    smoothing = obj.modifiers.new(name="Smooth", type="SMOOTH")
    smoothing.iterations = iterations
    smoothing.factor = factor
    bpy.ops.object.shade_smooth()
    recalculate_normals(obj)


def recalculate_normals(obj):
    bpy.ops.object.select_all(action="DESELECT")
    select(obj, True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.editmode_toggle()


def prepare_rendering():
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 256
    # Reenable this when using a different blender version
    bpy.context.scene.cycles.use_denoising = False
    # bpy.context.scene.cycles.denoiser = "OPTIX"
    bpy.context.scene.render.resolution_x = 800
    bpy.context.scene.render.resolution_y = 2000
    bpy.context.scene.render.resolution_percentage = 100


def set_background_to_transparent():
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
    bpy.context.scene.render.film_transparent = True


def set_up_default_preferences():
    # set viewport background to white
    bpy.context.preferences.themes["Default"].view_3d.space.gradients.high_gradient = (1, 1, 1)
    # turn of splash screen
    bpy.context.preferences.view.show_splash = False
    # set line thickness
    # bpy.context.preferences.ui_line_width = "thick"


def add_light_plane():
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children["Collection"]
    bpy.ops.mesh.primitive_plane_add()
    plane = bpy.data.objects["Plane"]
    plane.name = "LightPlane"
    # hide(plane, viewport=True)
    plane.scale = (22, 80, 1)
    plane.location = (0, 20, 40)

    mat_name = "LightMat"
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    mat.node_tree.nodes.new(type="ShaderNodeEmission")
    mat.node_tree.nodes["Emission"].inputs[1].default_value = 5
    inp = mat.node_tree.nodes["Material Output"].inputs["Surface"]
    outp = mat.node_tree.nodes["Emission"].outputs["Emission"]
    mat.node_tree.links.new(inp, outp)
    plane.active_material = mat


def remove_default_objects():
    for object_name in ["Cube", "Lamp", "Light"]:
        if object_name in bpy.data.objects:
            current_object = bpy.data.objects[object_name]
            bpy.data.objects.remove(current_object)


def deg2rad(angle_degrees: float) -> float:
    return angle_degrees * math.pi / 180


def rad2deg(angle_radians: float) -> float:
    return angle_radians / math.pi * 180


def set_up_camera():
    cor_camera = bpy.data.objects["Camera"]
    cor_camera.location = (24, 0, 0)
    cor_camera.rotation_euler = (deg2rad(90), 0, deg2rad(90))
    cor_camera.name = "CoronalCamera"
    cameras.append(cor_camera)
    # hide(camera, viewport=True)

    bpy.ops.object.camera_add(location=(17, 17, 0))
    corsag_camera = bpy.context.active_object
    corsag_camera.rotation_euler = (deg2rad(90), 0, deg2rad(135))
    corsag_camera.name = "CoroSagittalCamera"
    cameras.append(corsag_camera)

    bpy.ops.object.camera_add(location=(0, 24, 0))
    sag_camera = bpy.context.active_object
    sag_camera.rotation_euler = (deg2rad(90), 0, deg2rad(180))
    sag_camera.name = "SagittalCamera"
    cameras.append(sag_camera)

    bpy.context.scene.camera = corsag_camera


class AddPlaneAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        def parse_plane(plane_as_str: str):
            """
            >>> parse_plane("Th12=[(1, 2, 3), (2, 3, 4), (2.2, 4.2, 5.2), (1e2, -2e1, -2.3e-4)]")
            ('Th12', [(1.0, 2.0, 3.0), (2.0, 3.0, 4.0), (2.2, 4.2, 5.2), (100.0, -20.0, -0.00023)])
            """
            import re

            plane_as_str_no_whitespace = re.sub(r"\s+", "", plane_as_str)
            match = re.fullmatch(r"(\w+)=\[(.*)]", plane_as_str_no_whitespace)
            if match:
                name = match.group(1)
                # number = r"[+-]?((\d+\.?\d*)?|\.\d+)([eE]-?\d+)?"
                number = r"[-+0-9e.]+"
                tuple_regex = rf"\(({number},{number},{number})\)"
                coordinates = re.findall(tuple_regex, match.group(2))
                coordinates = [tuple(map(float, coord.split(","))) for coord in coordinates]
                return name, coordinates
            return None

        if not hasattr(args, "planes"):
            setattr(args, "planes", [])
        for value in values:
            plane_tuple = parse_plane(value)
            if plane_tuple:
                args.planes.append(plane_tuple)


class AddObjectAction(argparse.Action):
    def __init__(self, smooth=False, *args, **kwargs):
        self.smooth = smooth
        super().__init__(*args, **kwargs)

    def __call__(self, parser, args, values, option_string=None, smooth=False):
        if not hasattr(args, "objects"):
            setattr(args, "objects", [])
        for path in values:
            obj = load_obj(path)
            if obj is not None:
                # obj.rotation_euler = (0, 0, 0)
                obj.show_name = True
                if self.smooth:
                    make_obj_smooth(obj)
                bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
                args.objects.append(obj)


def get_areas_by_type(context, type_):
    return [a for a in context.screen.areas if a.type == type_]


def show_names_in_current_screen():
    for view3d_area in get_areas_by_type(bpy.context, "VIEW_3D"):
        view_space = view3d_area.spaces[0]
        view_space.show_only_render = False
        view_space.show_floor = False
        view_space.show_axis_x = False
        view_space.show_text = False
        view_space.show_axis_y = False
        view_space.show_axis_z = False
        view_space.cursor_location = (0, 0, 1000)


def make_intersection(for_obj, intersecting_obj):
    modifier_name = f"{for_obj.name}_{intersecting_obj.name}_intersect_modifier"
    mask_modifier = for_obj.modifiers.new(type="BOOLEAN", name=modifier_name)
    mask_modifier.object = intersecting_obj
    mask_modifier.operation = "INTERSECT"
    bpy.context.view_layer.objects.active = for_obj
    # Not needed because on export apply_modifiers=True is used, however, it is necessary
    # so that normals can be recalculated.
    bpy.ops.object.modifier_apply(apply_as="DATA", modifier=modifier_name)
    recalculate_normals(for_obj)


def apply_boolean_mask_to_spine_object():
    for obj_name in ["Spinal_canal", "Vertebrae", "Intervertebral_discs"]:
        obj = bpy.data.objects.get(obj_name, None)
        mask_cube = bpy.data.objects.get("AxisParallelCuboid1", None)
        if mask_cube is None or obj is None:
            continue
        make_intersection(obj, mask_cube)


class AddCollectionAction(argparse.Action):
    def __init__(self, hidden=False, *args, **kwargs):
        self.hidden = hidden
        super().__init__(*args, **kwargs)

    def __call__(self, parser, args, values, option_string=None):
        name = "collections" + ("_hidden" if self.hidden else "")
        if not hasattr(args, name):
            setattr(args, name, [])
        for value in values:
            collection = bpy.data.collections.new(value)
            bpy.context.scene.collection.children.link(collection)
            bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[value]
            getattr(args, name).append(collection)


class ExportToAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        if not hasattr(args, "export_to"):
            setattr(args, "export_to", [])
        args.export_to.append(Path(values))


class RenderToAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        if not hasattr(args, "render_to"):
            setattr(args, "render_to", [])
        args.render_to.append(Path(values))


def disable_hidden_collections(parsed_args):
    if hasattr(parsed_args, "collections_hidden"):
        pass
        # view_layer = bpy.context.scene.view_layers["View Layer"]
        # hidden_collection_names = set(col.name for col in parsed_args.collections_hidden) | {"Collection"}
        # for col in view_layer.layer_collection.children:
        #     col.exclude = col.name in hidden_collection_names
        # Alternatively to hide monitor:
        #   bpy.data.collections[col.name].hide_viewport = True
        # Alternatively to hide in current viewport:
        #   bpy.context.view_layer.layer_collection.children[col.name].hide_viewport = True


def export_colladas(parsed_args):
    for path in parsed_args.export_to:
        print(path)
        bpy.ops.wm.collada_export(filepath=path, apply_modifiers=True)


def render_and_save(parsed_args):
    for camera in cameras:
        bpy.context.scene.camera = camera
        if not hasattr(parsed_args, "render_to"):
            return
        bpy.ops.render.render()
        render_image = bpy.data.images["Render Result"]
        for path in parsed_args.render_to:
            render_image.save_render(filepath=str(path.with_stem(path.stem + camera.name)))


def main():
    remove_default_objects()
    set_up_camera()
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--object", nargs="+", action=AddObjectAction)
    arg_parser.add_argument("--object-smooth", nargs="+", action=partial(AddObjectAction, smooth=True))
    arg_parser.add_argument("--collection", nargs="+", action=AddCollectionAction)
    arg_parser.add_argument("--collection-hidden", nargs="+", action=partial(AddCollectionAction, hidden=True))
    # arg_parser.add_argument("--plane", nargs="+", action=AddPlaneAction)
    arg_parser.add_argument("--export-to", default=[], action=ExportToAction)
    arg_parser.add_argument("--render-to", default=[], action=RenderToAction)
    parsed_args = arg_parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    prepare_rendering()
    set_background_to_transparent()
    set_up_default_preferences()
    add_light_plane()
    apply_boolean_mask_to_spine_object()
    disable_hidden_collections(parsed_args)
    print(parsed_args)
    export_colladas(parsed_args)
    render_and_save(parsed_args)
    # show_names_in_current_screen()
    # bpy.ops.outliner.show_one_level(open=False)


if __name__ == "__main__":
    # Internal Blender import, not a package. Import here so that doctests can still be run
    # as modules. Doctests run as doctests are not considered to be main, and therefore this import is added here.
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    import bpy

    main()
