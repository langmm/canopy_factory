import os
import pdb
import numpy as np
import itertools
import pprint
import copy
from datetime import datetime
from collections import OrderedDict
from yggdrasil_rapidjson import units
from yggdrasil_rapidjson.geometry import Ply as PlyDict
from yggdrasil_rapidjson.geometry import ObjWavefront as ObjDict
from canopy_factory import utils, arguments
from canopy_factory.utils import (
    cfg, RegisteredClassBase, get_class_registry,
    parse_quantity, parse_axis,
    cached_property, cached_args_property, readonly_cached_args_property,
    # Geometry
    scene2geom,
)
from canopy_factory.cli import (
    TaskBase, TemporalTaskBase,
    OptimizationTaskBase,
)
from canopy_factory.crops import GenerateTask, LayoutTask
from canopy_factory import light_sources


_query_options_adm = [
    'plantids', 'componentids',
]
_query_options_phy = [
    'flux_density', 'flux', 'hits', 'areas',
]
_query_options = _query_options_phy + _query_options_adm
_query_options_calc_prefix = [
    'total_', 'average_', 'scene_average_',
]
_query_options_calc = [
    prefix + q for prefix, q in
    itertools.product(_query_options_calc_prefix, _query_options_phy)
]


def generate_rays(ray_origins, ray_directions,
                  geom_format='obj', ray_color=(1.0, 0.0, 0.0),
                  ray_width=1.0, ray_length=10.0, arrow_width=2.0,
                  verbose=False):
    r"""Generate a set of rays for demonstration purposes.

    Args:
        ray_origins (np.ndarray): Positions that rays should start from.
        ray_directions (np.ndarray): Vectors describing how rays should
            point.
        geom_format (str, optional): Format that the rays geometries
            should be returned as.
        ray_color (tuple, optional): Set of RGB values in range 0.0 to 1.0
            designating the color that rays should be.
        ray_width (float, optional): Width of the ray stems.
        ray_length (float, optional): Length of the ray stems. This can
            also be provided as an array for each individual ray.
        arrow_width (float, optional): Width of the ray head.
        verbose (bool, optional): If True, log messages will be displayed
            for the generation process.

    """
    from openalea.lpy import Lsystem
    param = {
        'ORIGINS': ray_origins.astype(np.float64),
        'DIRECTIONS': ray_directions.astype(np.float64),
        'RAY_WIDTH': ray_width,
        'RAY_LENGTH': ray_length,
        'ARROW_WIDTH': arrow_width,
    }
    lsys = Lsystem(cfg['directories']['lpy'], param)
    tree = lsys.axiom
    for i in range(2):
        tree = lsys.iterate(tree, 1)
    scene = lsys.sceneInterpretation(tree)
    mesh = scene2geom(
        scene, geom_format,
        color=ray_color, verbose=verbose,
    )
    if verbose:
        print('Finished generating rays')
    return mesh


class SceneModel(RegisteredClassBase):
    r"""Container for scene properties.

    Args:
        mins (np.ndarray): Minimum bounds of scene in each dimension.
        maxs (np.ndarray): Maximum bounds of scene in each dimension.

    """

    def __init__(self, mins, maxs):
        self.mins = mins
        self.maxs = maxs

    @cached_property
    def limits(self):
        r"""np.ndarray: Corners of a box containing the scene."""
        limits = np.vstack([self.mins, self.maxs])
        xx, yy, zz = np.meshgrid(limits[:, 0], limits[:, 1], limits[:, 2])
        out = np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T
        return out

    @cached_args_property
    def center(self):
        r"""np.ndarray: Coordinates of the scene's center."""
        return (self.maxs + self.mins) / 2

    @cached_args_property
    def dim(self):
        r"""np.ndarray: Scene's dimensions in each direction."""
        return (self.maxs - self.mins)


class RayTracerBase(RegisteredClassBase):
    r"""Base class for ray tracers."""

    _registry_key = 'raytracer'
    _area_min = np.finfo(np.float32).resolution

    def __init__(self, args, mesh, geometryids=None):
        super(RayTracerBase, self).__init__()
        if args.output_generate.generated:
            assert geometryids is not None
        self.args = args
        self.mesh = mesh
        self.geometryids = geometryids
        self.verbose = self.args.verbose
        self.mesh_dict = utils.get_mesh_dict(self.mesh)
        self.areas = np.array(self.mesh.areas)
        self.area_mask = (self.areas > self._area_min)
        if self.args.include_units:
            self.areas = parse_quantity(
                self.areas, self.args.mesh_units**2)
        self.plants = {k: self.mesh_item(k) for k in self.mesh_keys()}
        for plantid in self.plants.keys():
            self.plants[plantid] = self.select_faces(
                self.plants[plantid], self.area_mask,
                dont_prune_vertices=True,
            )
        self.log(f'Creating scene with up = {self.up}, '
                 f'north = {self.north}, '
                 f'east = {self.east}, ground = {self.ground}')

    @cached_property
    def periodic_canopy(self):
        r"""str: Periodic canopy type."""
        if self.args.virtual_canopy and not self.args.periodic_canopy:
            return 'scene'
        return self.args.periodic_canopy

    @cached_property
    def periodic_canopy_count_array(self):
        r"""np.ndarray: Array specifying how many times the canopy
        should be replicated in each direction."""
        out = self.args.periodic_canopy_count_array
        if self.args.virtual_canopy:
            out = copy.deepcopy(out)
            out[0] *= self.args.virtual_nrows
            out[1] *= self.args.virtual_ncols
        return out

    @cached_property
    def geometryid_order(self):
        r"""list: Set of IDs that should be used to split the geometry
        into parts."""
        out = []
        if self.geometryids is not None:
            if self.args.separate_plants:
                out.append('plantids')
            if self.args.separate_components:
                out.append('componentids')
        return out

    def mesh_keys(self):
        r"""Get the set of keys splitting up the mesh geometry into
        parts.

        Returns:
            list: Keys.

        """
        keys = [list(np.unique(self.geometryids[k])) for k in
                self.geometryid_order]
        if not keys:
            return [None]
        return [tuple(x) for x in itertools.product(*keys)]

    def mesh_item(self, key):
        r"""Get the portion of the mesh corresponding to a key specifying
        which part of the geometry should be selected.

        Args:
            key (tuple, None): Key specifying a part of the geometry.

        Returns:
            dict: Mesh dictionary for selected part of the geometry.

        """
        if key is None:
            return self.mesh_dict
        assert len(key) == len(self.geometryid_order)
        idx = np.zeros(self.areas.shape, dtype=np.uint8)
        for k, v in zip(self.geometryid_order, key):
            idx += (self.geometryids[k] == v)
        return self.select_faces(self.mesh_dict, (idx == len(key)),
                                 continuous=(key == ('plantids', )))

    def parse_axis(self, x):
        r"""Parse axis values that specify direction relative to the
        scene.

        Args:
            x (str): Input string.

        Returns:
            np.ndarray: Axis vector.

        """
        directions = OrderedDict([
            ('up', self.up),
            ('down', -self.up),
            ('north', self.north),
            ('south', -self.north),
            ('east', self.east),
            ('west', -self.east),
        ])
        if isinstance(x, str):
            if x in directions:
                return directions[x]
            composite = []
            xpartial = x
            for k, v in directions.items():
                if xpartial.startswith(k):
                    composite.append(v)
                    xpartial = xpartial.split(k, 1)[-1]
            if composite and not xpartial:
                out = np.mean(np.vstack(composite), axis=0)
                assert len(out) == 3
                out /= np.linalg.norm(out)
                return out
        return parse_axis(x)

    @classmethod
    def assign_face_data(cls, idx, dst, src):
        r"""Assign face values to a destination array based on the
        supplied index.

        Args:
            idx (dict): Dictionary of indices created when selecting a
                subset of a mesh.
            dst (np.ndarray): Array that values should be copied into.
            src (np.ndarray): Array of face values for the current
                selection of faces.

        """
        idx_chain = []
        iidx = idx
        while iidx is not None:
            idx_chain.append(iidx['face'])
            iidx = iidx.get('parent', None)
        idx_dst = np.arange(len(dst), dtype=np.int32)
        for iidx in idx_chain[::-1]:
            assert idx_dst.shape == iidx.shape
            idx_dst = idx_dst[iidx]
        assert idx_dst.shape == src.shape
        dst[idx_dst] = src

    @classmethod
    def select_vertices(cls, mesh_dict, cond, continuous=False):
        r"""Select a subset of the vertices in a mesh.

        Args:
            mesh_dict (dict): Dictionary of mesh properties.
            cond (np.ndarray): Condition that should be used to select
                the vertices.
            continuous (bool, optional): If True, the faces and vertices
                for which the condition is True are continuous.

        Returns:
            dict: Dictionary of mesh properties with only the selected
                vertices.

        """
        if ((len(cond) != mesh_dict['vertex'].shape[0]
             and 'idx' in mesh_dict)):
            cond = cond[mesh_dict['idx']['vertex']]
        assert len(cond) == mesh_dict['vertex'].shape[0]
        # This verifies that the condition is the same for all vertices
        # in each face.
        # for i, face in enumerate(mesh_dict['face']):
        #     cond_face = cond[face]
        #     assert all(cond_face == cond_face[0])
        cond_face = cond[mesh_dict['face'][:, 0]]
        return cls.select_faces(mesh_dict, cond_face,
                                continuous=continuous)

    @classmethod
    def select_faces(cls, mesh_dict, cond, continuous=False,
                     dont_prune_vertices=False):
        r"""Select a subset of the faces in a mesh.

        Args:
            mesh_dict (dict): Dictionary of mesh properties.
            cond (np.ndarray): Condition that should be used to select
                the faces.
            continuous (bool, optional): If True, the faces and vertices
                for which the condition is True are continuous.
            dont_prune_vertices (bool, optional): If True, vertices that
                are not selected are not removed.

        Returns:
            dict: Dictionary of mesh properties with only the selected
                faces.

        """
        out_class = None
        if isinstance(mesh_dict, (ObjDict, PlyDict)):
            out_class = type(mesh_dict)
            mesh = mesh_dict
            mesh_dict = utils.get_mesh_dict(mesh)
        if len(cond) != mesh_dict['face'].shape[0] and 'idx' in mesh_dict:
            cond = cond[mesh_dict['idx']['face']]
        assert len(cond) == mesh_dict['face'].shape[0]
        if np.logical_not(cond).sum() == 0:
            out = mesh_dict
            if out_class is not None:
                out.pop('idx')
                out = out_class.from_dict(out)
            return out
        out = {
            'idx': {'face': cond},
            'face': mesh_dict['face'][cond, :],
        }
        if 'idx' in mesh_dict:
            out['idx']['parent'] = mesh_dict['idx']
        nvert = mesh_dict['vertex'].shape[0]
        out['idx']['vertex'] = np.zeros((nvert, ), dtype=bool)
        if dont_prune_vertices:
            out['idx']['vertex'][:] = True
        else:
            out['idx']['vertex'][np.unique(out['face'])] = True
            if continuous:
                # This is for double checking that vertices are continuous
                # idx_vert = out['idx']['vertex']
                # idx_vert2 = out['idx']['vertex'][1:]
                # idx_vert1 = out['idx']['vertex'][:-1]
                # assert all((idx_vert2 - idx_vert1) == 1)
                out['face'] -= out['face'].min()
            else:
                idx_remove = np.where(
                    np.logical_not(out['idx']['vertex']))[0]
                for idx in idx_remove[::-1]:
                    out['face'][out['face'] > idx] -= 1
        for k in ['vertex', 'vertex_colors']:
            if k not in mesh_dict:
                continue
            out[k] = mesh_dict[k][out['idx']['vertex'], :]
        if out_class is not None:
            out.pop('idx')
            out = out_class.from_dict(out)
        return out

    @cached_args_property
    def up(self):
        r"""np.ndarray: Vector direction of up in the scene."""
        return self.args.axis_up.astype("f4")

    @cached_args_property
    def north(self):
        r"""np.ndarray: Vector direction of north in the scene."""
        return self.args.axis_north.astype("f4")

    @cached_args_property
    def east(self):
        r"""np.ndarray: Vector direction of east in the scene."""
        return np.cross(self.north, self.up)

    @cached_args_property
    def virtual_shifts(self):
        r"""np.ndarray: Shifts for positions of virtual plants."""
        if not self.args.virtual_canopy:
            return None
        return LayoutTask.get_periodic_shifts(
            self.args.periodic_period,
            self.args.periodic_direction,
            np.array([
                self.args.virtual_nrows,
                self.args.virtual_ncols,
                0
            ], 'i4'),
            include_origin=True,
        )

    @cached_args_property
    def real_scene_model(self):
        r"""SceneModel: Scene properties."""
        mins = None
        maxs = None
        if getattr(self.args, 'scene_mins', None) is not None:
            mins = self.args.scene_mins
        elif self.mesh_dict['vertex'].shape[0] == 0:
            mins = np.zeros((3, ), dtype=self.mesh_dict['vertex'].dtype)
        else:
            mins = self.mesh_dict['vertex'].min(axis=0)
        if getattr(self.args, 'scene_maxs', None) is not None:
            maxs = self.args.scene_maxs
        elif self.mesh_dict['vertex'].shape[0] == 0:
            maxs = np.zeros((3, ), dtype=self.mesh_dict['vertex'].dtype)
        else:
            maxs = self.mesh_dict['vertex'].max(axis=0)
        return SceneModel(mins, maxs)

    @cached_args_property
    def virtual_scene_model(self):
        r"""SceneModel: Virtual scene properties."""
        if not self.args.virtual_canopy:
            return self.real_scene_model
        mins = self.real_scene_model.mins
        maxs = self.real_scene_model.maxs
        if getattr(self.args, 'scene_mins', None) is None:
            mins = (mins + self.virtual_shifts).min(axis=0)
        if getattr(self.args, 'scene_maxs', None) is None:
            maxs = (maxs + self.virtual_shifts).max(axis=0)
        return SceneModel(mins, maxs)

    @cached_args_property
    def ground(self):
        r"""np.ndarray: Coordinates of the scene center on the ground."""
        return (
            np.dot(self.real_scene_model.dim / 2, self.north) * self.north
            + np.dot(self.real_scene_model.dim / 2, self.east) * self.east
            + self.args.ground_height.value * self.up
        )

    @cached_args_property
    def camera_up(self):
        r"""np.ndarray: Unit vector in the camera's up direction."""
        if self.args.camera_up is not None:
            return self.parse_axis(self.args.camera_up)
        vadj = -self.camera_direction
        vadjup = np.dot(vadj, self.up)
        if vadjup == 0.0:
            out = self.up
        else:
            angle = np.arccos(vadjup)
            vhyp = self.up / np.cos(angle)
            out = vhyp - vadj
        out /= np.linalg.norm(out)
        return out

    @cached_args_property
    def camera_direction(self):
        r"""np.ndarray: Unit vector for camera's pointing direction."""
        if self.args.camera_direction is not None:
            return self.parse_axis(self.args.camera_direction)
        if self.args.camera_location is None:
            out = self.parse_axis('downsoutheast')
        else:
            out = self.virtual_scene_model.center - self.camera_location
        out /= np.linalg.norm(out)
        return out

    @cached_args_property
    def camera_right(self):
        r"""np.ndarray: Unit vector for camera's right direction."""
        return np.cross(self.camera_direction, self.camera_up)

    @cached_args_property
    def camera_location(self):
        r"""np.ndarray: Coordinates of the camera."""
        if self.args.camera_location is not None:
            return self.parse_axis(self.args.camera_location)
        fov_width = np.max(np.abs(np.dot(
            self.virtual_scene_model.limits - self.virtual_scene_model.center,
            self.camera_right)))
        camera_distance = np.abs(
            fov_width / np.tan(self.args.camera_fov_width / 2.0))
        if camera_distance < self.clipping_distance.value:
            camera_distance = self.clipping_distance.value
        out = (
            self.virtual_scene_model.center
            - (camera_distance * self.camera_direction)
        )
        if isinstance(out, units.QuantityArray) and out.is_dimensionless():
            out = np.array(out)
        return out

    @cached_args_property
    def camera_distance(self):
        r"""float: Distance between the camera and the scene scenter."""
        return units.Quantity(
            np.linalg.norm(
                self.virtual_scene_model.center - self.camera_location),
            self.args.mesh_units)

    @cached_args_property
    def image_width(self):
        r"""float: Image width."""
        if self.args.image_width is not None:
            return self.args.image_width
        if self.args.camera_type == 'projection':
            return 2.0 * self.image_distance * np.abs(np.tan(
                self.args.camera_fov_width / 2.0))
        elif self.args.camera_type == 'orthographic':
            return self.camera_scene_dims[0]
        else:
            raise NotImplementedError(f'Default image width for '
                                      f'{self.args.camera_type} camera')

    @cached_args_property
    def image_height(self):
        r"""float: Image height."""
        if self.args.image_height is not None:
            return self.args.image_height
        if self.args.camera_type == 'projection':
            return 2.0 * self.image_distance * np.abs(np.tan(
                self.args.camera_fov_height / 2.0))
        elif self.args.camera_type == 'orthographic':
            return self.camera_scene_dims[1]
        else:
            raise NotImplementedError(f'Default image height for '
                                      f'{self.args.camera_type} camera')

    @cached_args_property
    def camera_scene_dims(self):
        r"""np.ndarray: Scene dimensions parallel to the image plant."""
        return np.array([
            2 * np.max(np.abs(np.dot(
                self.virtual_scene_model.limits - self.camera_location,
                self.camera_right))),
            2 * np.max(np.abs(np.dot(
                self.virtual_scene_model.limits - self.camera_location,
                self.camera_up))),
            2 * np.max(np.abs(np.dot(
                self.virtual_scene_model.limits - self.camera_location,
                self.camera_direction))),
        ])

    @cached_args_property
    def clipping_distance(self):
        r"""float: Maximum distance of any scene limits from the scene
        center along the camera line-of-sight."""
        return units.Quantity(
            np.max(np.abs(np.dot(
                self.virtual_scene_model.limits
                - self.virtual_scene_model.center,
                self.camera_direction))),
            self.args.mesh_units)

    @cached_args_property
    def image_distance(self):
        r"""float: Distance of the image plane from the camera."""
        if self.args.camera_type == 'projection':
            if self.args.image_width is not None:
                return np.abs(
                    (self.image_width / 2.0)
                    / np.tan(self.args.camera_fov_width / 2.0))
            elif self.args.image_height is not None:
                return np.abs(
                    (self.image_height / 2.0)
                    / np.tan(self.args.camera_fov_height / 2.0))
            out = self.camera_distance - self.clipping_distance
            if out.value < 0:
                out.value = 0.0
            return out
        elif self.args.camera_type == 'orthographic':
            return units.Quantity(0.0, self.args.mesh_units)
        else:
            raise NotImplementedError(f'Default image distance for '
                                      f'{self.args.camera_type} camera')

    @cached_args_property
    def image_center(self):
        r"""np.ndarray: Coordinates of the image's center."""
        return (
            self.camera_location +
            (self.image_distance.value * self.camera_direction)
        )

    @cached_args_property
    def resolution(self):
        r"""float: Number of pixels per cm that image should be rendered
        will."""
        if self.args.resolution is not None:
            return self.args.resolution
        elif self.args.image_nx is not None:
            return self.image_nx / self.image_width
        return self.image_ny / self.image_height

    @cached_args_property
    def image_nx(self):
        r"""int: Number of pixels in the x direction."""
        if ((self.args.resolution is not None
             or self.args.image_nx is None)):
            return int(self.image_width * self.resolution)
        return self.args.image_nx

    @cached_args_property
    def image_ny(self):
        r"""int: Number of pixels in the y direction."""
        if ((self.args.resolution is not None
             or (self.args.image_ny is None
                 and self.args.image_nx is not None))):
            return int(self.image_height * self.resolution)
        elif self.args.image_ny is not None:
            return self.args.image_ny
        return 2048

    @property
    def ray_origins(self):
        r"""np.ndarray: Coordinates of ray origins."""
        raise NotImplementedError

    @property
    def ray_directions(self):
        r"""np.ndarray: Ray directions."""
        raise NotImplementedError

    @property
    def ray_lengths(self):
        r"""np.ndarray: Ray lengths."""
        raise NotImplementedError

    def raytrace(self):
        r"""Run the ray tracer and get values for each face.

        Returns:
            np.ndarray: Ray tracer results for each face.

        """
        raise NotImplementedError

    def render(self, values):
        r"""Image the scene.

        Args:
            values (np.ndarray): Values on each face that should be used
                when imaging the scene.

        Returns:
            np.ndarray: Ray tracer results for each pixel.

        """
        raise NotImplementedError

    def face2vertex(self, face_scalar, method='average'):
        r"""Convert an array of scalars for each face to an array of
        scalars for each vertex.

        Args:
            face_scalar (np.ndarray): Array of scalars for each face.
            method (str, optional): Method to use to map from face values
                to vertex values.
                    'average': Average over the values for each face that
                        vertices are part of.
                    'deposit': Split the values for each face amongst its
                        vertices additively.

        Returns:
            np.ndarray: Array of scalars for each vertex.

        """
        faces = self.mesh_dict['face'][self.area_mask, :]
        face_scalar = face_scalar[self.area_mask]
        if method == 'deposit':
            face_scalar /= faces.shape[1]
        vertex_scalar = np.zeros((self.mesh_dict['vertex'].shape[0], ))
        try:
            face_scalar = np.tile(face_scalar, (faces.shape[1], 1)).T
        except units.UnitsError:
            face_scalar = units.QuantityArray(
                np.tile(face_scalar.value, (faces.shape[1], 1)).T,
                face_scalar.units)
        assert face_scalar.shape == faces.shape
        for idx, scalar in zip(faces.flatten(), face_scalar.flatten()):
            vertex_scalar[idx] += scalar
        if method == 'average':
            unique, vertex_counts = np.unique(faces, return_counts=True)
            vertex_scalar[unique] /= vertex_counts
        return vertex_scalar


class HothouseRayTracer(RayTracerBase):

    _name = 'hothouse'

    @cached_args_property
    def scene(self):
        r"""hothouse.scene.Scene: Scene containing geometry."""
        from hothouse.plant_model import PlantModel
        kws = {}
        if self.periodic_canopy == 'scene':
            from hothouse.scene import PeriodicScene as Scene
            kws.update(
                period=self.args.periodic_period.astype('f4'),
                direction=self.args.periodic_direction.astype('f4'),
                count=self.periodic_canopy_count_array,
            )
        else:
            from hothouse.scene import Scene
        out = Scene(
            ground=self.ground, up=self.up, north=self.north, **kws
        )
        for plantid, mesh_dict in self.plants.items():
            triangles = []
            for face in mesh_dict['face']:
                triangles.append(mesh_dict['vertex'][face, :])
            if triangles:
                triangles = np.array(triangles)
                plant = PlantModel(
                    vertices=mesh_dict['vertex'].astype('f4'),
                    indices=mesh_dict['face'].astype('i4'),
                    attributes=mesh_dict['vertex_colors'].astype('f4'),
                    triangles=triangles.astype('f4'),
                )
                out.add_component(plant)
        return out

    @cached_args_property
    def camera_blaster(self):
        r"""hothouse.blaster.OrthographicRayBlaster: Blaster for camera."""
        from hothouse.blaster import (
            ProjectionRayBlaster, OrthographicRayBlaster,
            SphericalRayBlaster)
        camera_classes = {
            'projection': ProjectionRayBlaster,
            'orthographic': OrthographicRayBlaster,
            'spherical': SphericalRayBlaster,
        }
        kws = {}
        if self.args.camera_type == 'projection':
            kws['fov_width'] = self.args.camera_fov_width
            kws['fov_height'] = self.args.camera_fov_height
        rbcls = camera_classes[self.args.camera_type]
        assert self.image_width.value > 0
        camera_blaster = rbcls(
            center=self.image_center.astype("f4"),
            forward=self.camera_direction.astype("f4"),
            up=self.camera_up.astype("f4"),
            width=self.image_width,
            height=self.image_height,
            nx=self.image_nx,
            ny=self.image_ny,
            **kws
        )
        return camera_blaster

    @cached_args_property
    def solar_model(self):
        r"""SolarModel: Model for the sun using pvlib."""
        return self.args.time.solar_model

    @cached_args_property
    def point_source_blaster(self):
        r"""hothouse.blaster.SolarBlaster: Blaster for a point source."""
        from hothouse.blaster import SphericalRayBlaster
        assert self.periodic_canopy != 'rays'
        direct_ppfd = self.args.light_intensity_direct
        diffuse_ppfd = self.args.light_intensity_diffuse
        self.log(f"Total PPFD"
                 f"\n   direct = {direct_ppfd}"
                 f"\n   diffuse = {diffuse_ppfd}",
                 force=True)
        kws = {}
        if self.args.light_source == 'point_grid':
            kws.update(
                period=self.args.light_grid_period,
                periodic_direction=self.args.light_grid_periodic_direction,
                periodic_count=self.args.light_grid_periodic_count_array,
            )
        return SphericalRayBlaster(
            center=self.args.light_source_location,
            forward=self.args.light_source_direction,  # Default to down
            fov_width=self.args.light_source_fov,  # Default to 180
            fov_height=self.args.light_source_fov,
            direct_ppfd=direct_ppfd,
            diffuse_ppfd=diffuse_ppfd,
            nx=self.args.nrays, ny=self.args.nrays,
            multibounce=self.args.multibounce, **kws
        )

    @cached_args_property
    def solar_blaster(self):
        r"""hothouse.blaster.SolarBlaster: Blaster for sun."""
        # TODO: Add units to parser
        self.log(f"Total PPFD"
                 f"\n   direct = {self.solar_model.ppfd_direct}"
                 f"\n   diffuse = {self.solar_model.ppfd_diffuse}",
                 force=True)
        kws = {}
        if self.periodic_canopy == 'rays':
            kws.update(
                period=self.args.periodic_period.astype('f4'),
                periodic_direction=self.args.periodic_direction.astype('f4'),
                periodic_count=self.periodic_canopy_count_array,
            )
        return self.scene.get_sun_blaster(
            self.solar_model.latitude.value,
            self.solar_model.longitude.value,
            self.solar_model.time,
            direct_ppfd=self.solar_model.ppfd_direct.value[0],
            diffuse_ppfd=self.solar_model.ppfd_diffuse.value[0],
            solar_altitude=self.solar_model.apparent_elevation,
            solar_azimuth=self.solar_model.azimuth,
            nx=self.args.nrays, ny=self.args.nrays,
            multibounce=self.args.multibounce, **kws
        )

    @property
    def ray_origins(self):
        r"""np.ndarray: Coordinates of ray origins."""
        return self.ray_properties[0]

    @property
    def ray_directions(self):
        r"""np.ndarray: Ray directions."""
        return self.ray_properties[1]

    @property
    def ray_lengths(self):
        r"""np.ndarray: Ray lengths."""
        return self.ray_properties[2]

    @cached_args_property
    def ray_properties(self):
        r"""tuple: Ray properties."""
        rb = self.scene.get_sun_blaster(
            self.solar_model.latitude, self.solar_model.longitude,
            self.solar_model.time,
            solar_altitude=self.solar_model.apparent_elevation,
            solar_azimuth=self.solar_model.azimuth,
            nx=10, ny=10, multibounce=self.args.multibounce,
        )
        ray_origins = rb.origins
        ray_directions = rb.directions
        ray_lengths = self.args.ray_length.value
        if ray_lengths < 0:
            ray_length0 = -ray_lengths
            ray_lengths = rb.compute_distance(self.scene)
            idx_max = (ray_lengths >= max(ray_lengths))
            ray_lengths[idx_max] = ray_length0
        return (ray_origins, ray_directions, ray_lengths)

    def raytrace(self):
        r"""Run the ray tracer and get values for each face.

        Returns:
            np.ndarray: Ray tracer results for each face.

        """
        if self.args.query == 'areas':
            return self.areas
        elif self.geometryids and self.args.query in self.geometryids:
            return self.geometryids[self.args.query]
        values = np.zeros((self.mesh_dict['face'].shape[0], ), np.float64)
        if values.shape[0] == 0:
            return values
        self.log(f'Running ray tracer to get {self.args.query} for '
                 f't = {self.args.time.time}, age = {self.args.time.age} '
                 f'({self.args.age.value}) with '
                 f'light direction: {self.solar_blaster.forward}',
                 border=True, force=True)
        component_values = None
        value_units = None
        if self.args.query == 'flux_density':
            component_values = self.scene.compute_flux_density(
                self.solar_blaster,
                any_direction=self.args.any_direction,
            )
            if self.args.include_units:
                value_units = self.solar_model.ppfd_direct.units
        elif self.args.query == 'hits':
            component_values = self.scene.compute_hit_count(
                self.solar_blaster)
        else:
            raise ValueError(f"Unsupported ray tracer query "
                             f"\"{self.args.query}\"")
        for i, k in enumerate(self.mesh_keys()):
            v = component_values[i]
            self.assign_face_data(self.plants[k].get('idx', None),
                                  values, v)
        if value_units:
            values = parse_quantity(values, value_units)
        return values

    def render(self, values, value_miss=-1.0):
        r"""Image the scene.

        Args:
            values (np.ndarray): Values on each face that should be used
                when imaging the scene.
            value_miss (float, optional): Value that should be used for
                pixels that do not hit anything.

        Returns:
            np.ndarray: Ray tracer results for each pixel.

        """
        from hothouse.scene import PeriodicScene
        prev = None
        if isinstance(self.scene, PeriodicScene):
            prev = self.scene.buffer_as_primary
            self.scene.buffer_as_primary = True
        camera_hits = self.camera_blaster.compute_count(self.scene)
        if prev is not None:
            self.scene.buffer_as_primary = prev
        out = np.zeros(self.image_nx * self.image_ny, "f4")
        if isinstance(values, units.QuantityArray):
            out = units.QuantityArray(out, values.units)
            value_miss = parse_quantity(value_miss, values.units)
        for ci in range(len(self.scene.components)):
            idx_ci = np.where(camera_hits["geomID"] == ci)[0]
            hits = camera_hits["primID"][idx_ci]
            try:
                out[idx_ci[hits >= 0]] += values[hits[hits >= 0]]
            except TypeError:
                pdb.set_trace()
                raise
        out[camera_hits["primID"] < 0] = value_miss
        return out.reshape((self.image_nx, self.image_ny))


###################################################################
# TASKS
###################################################################

class RayTraceTask(TaskBase):
    r"""Class for running a ray tracer on a 3D canopy."""

    _name = 'raytrace'
    _output_info = {
        'raytrace': {
            'upstream': ['generate'],
            'ext': '.csv',
            'description': (
                'the query values for each face in the mesh'
            ),
        },
        'raytrace_limits': {
            'ext': '.json',
            'description': 'limits on raytrace query values',
            'optional': True,
        },
        'traced_mesh': {
            'description': (
                'the mesh with faces colored by a ray tracer result'
            ),
            'optional': True,
            'merge_all': True,
        },
    }
    _external_tasks = {
        GenerateTask: {
            'exclude': ['age'],
            'modifications': {
                'mesh_format': {
                    'help': (
                        'Format that provided \"mesh\" is in or the '
                        'format that the generate mesh should be in if '
                        '\"mesh\" is not provided. If \"--mesh-format\" '
                        'is not provided, the file extension will be '
                        'used to determine the format'
                    ),
                },
                'mesh_units': {
                    'help': (
                        'Units that the provided \"mesh\" is in or '
                        'the units the generated mesh should be in if '
                        '\"mesh\" is not provided'
                    ),
                },
                'canopy': {
                    'default': 'unique',
                    'append_choices': ['virtual'],
                },
            }
        },
        LayoutTask: {
            'include': [
                'periodic_canopy', 'periodic_canopy_count',
            ],
            'optional': True,
            'suffix_index': -1,
        },
    }
    _virtual_argument_names = [
        'canopy', 'row_spacing', 'plant_spacing',
        'nrows', 'ncols', 'plot_length', 'plot_width',
    ]
    _arguments = [
        # GenerateTask._arguments['crop'].copy(remove_class='task'),
        arguments.ArgumentDescriptionSet([
            (('--virtual-canopy', ), {
                'no_cli': True,
                'suffix_param': {'prefix': 'canopy', 'title': True},
            }),
            arguments.ArgumentDescriptionSet([
                (('--virtual-plot-length', ), {
                    'no_cli': True,
                }),
                (('--virtual-plot-width', ), {
                    'no_cli': True,
                }),
            ], name='virtual_plot_dimensions'),
            arguments.ArgumentDescriptionSet([
                (('--virtual-row-spacing', ), {
                    'no_cli': True,
                    'suffix_param': {
                        'noteq': parse_quantity(76.2, 'cm'),
                    },
                }),
                (('--virtual-plant-spacing', ), {
                    'no_cli': True,
                    'suffix_param': {
                        'noteq': parse_quantity(18.3, 'cm'),
                    },
                }),
            ], name='virtual_plant_spacing', suffix_param={
                'sep': 'x',
                'require_all': True,
            }),
            arguments.ArgumentDescriptionSet([
                (('--virtual-nrows', ), {
                    'no_cli': True,
                    'suffix_param': {'noteq': 4},
                }),
                (('--virtual-ncols', ), {
                    'no_cli': True,
                    'suffix_param': {'noteq': 10},
                }),
            ], name='virtual_plant_count', suffix_param={
                'sep': 'x',
                'require_all': True,
            }),
        ], name='virtual_canopy_properties', suffix_param={
            'cond': lambda x: bool(getattr(x, 'virtual_canopy', None)),
            'index': -1,
        }),
        arguments.ClassSubparserArgumentDescription(
            'light_source',
            suffix_param={'noteq': 'sun'},
        ),
        arguments.CompositeArgumentDescription(
            'ray_color', 'color',
            description=(
                ' that should be used for rays when "--show-rays" '
                'is passed'
            ),
            defaults={
                'color': 'red',
            },
        ),
        arguments.CompositeArgumentDescription(
            'highlight_color', 'color',
            description=(
                'that should be used for highlighted faces if '
                '"--highlight" is passed'
            ),
            defaults={
                'color': 'magenta',
            },
        ),
        (('--raytracer', ), {
            'type': str, 'default': 'hothouse',
            'choices': list(get_class_registry().keys('raytracer')),
            'help': 'Name of the ray tracer that should be used.',
            'suffix_param': {'noteq': 'hothouse'},
        }),
        (('--separate-plants', ), {
            'action': 'store_true',
            'help': 'Track plants as separate scene components.',
        }),
        (('--separate-components', ), {
            'action': 'store_true',
            'help': ('Track plant components (e.g. leaf, stem) as '
                     'separate scene components.'),
        }),
        (('--nrays', ), {
            'type': int, 'default': 512,
            'help': ('Number of rays that should be cast along each '
                     'dimension'),
            'suffix_param': {'cond': True},
        }),
        (('--multibounce', ), {
            'action': 'store_true',
            'help': ('Include multiple bounces when performing the '
                     'trace.'),
            'suffix_param': {'value': 'multibounce'},
        }),
        (('--any-direction', ), {
            'action': 'store_true',
            'help': ('Allow light to be deposited by the ray tracer '
                     'from any direction relative to the surface. If '
                     'not set, only ray intercepting a surface from '
                     'the \"top\" will be counted.'),
            'suffix_param': {'value': 'anydirection'},
        }),
        (('--query', ), {
            'type': str, 'choices': _query_options,
            'default': 'flux',
            'help': ('Name of the raytracer query result that should '
                     'be used to color the traced mesh if '
                     '"--output-traced-mesh" is passed. '
                     '\"flux\" uses the intercepted flux density for '
                     'each triangle in the mesh, \"hits\" uses '
                     'the number of rays that hit each face in the '
                     'mesh, \"areas\" uses the area of each face, '
                     '\"plantids\" uses the IDs of the plant each face '
                     'belongs to, and \"componentids\" uses the IDs '
                     'of the plant architecture component.'),
            'suffix_param': {
                'cond': True,
                'skip_outputs': ['raytrace', 'raytrace_limits'],
            },
        }),
        (('--show-rays', ), {
            'action': "store_true",
            'help': ('Show the rays in the generated mesh if '
                     '"--output-traced-mesh" is passed.'),
            'suffix_param': {
                'value': 'rays',
                'skip_outputs': ['raytrace', 'raytrace_limits'],
            },
        }),
        (('--highlight', ), {
            'type': str, 'choices': ['min', 'max'],
            'help': ('Highlight the face with the \"min\" or \"max\" '
                     'query value in the resulting (only valid if '
                     '"--output-traced-mesh" is passed).'),
            'suffix_param': {
                'prefix': 'highlight',
                'title': True,
                'skip_outputs': ['raytrace', 'raytrace_limits'],
            },
        }),
        (('--ray-width', ), {
            'default': 1.0, 'units': 'cm',
            'units_arg': 'mesh_units',
            'help': 'Width of rays drawn when "--show-rays" is passed.',
        }),
        (('--ray-length', ), {
            'default': 10.0, 'units': 'cm',
            'units_arg': 'mesh_units',
            'help': ('Length of rays drawn when "--show-rays" is '
                     'passed. A negative value will cause the distance '
                     'to the scene to be used for the ray length.'),
        }),
        (('--arrow-width', ), {
            'default': 2.0, 'units': 'cm',
            'units_arg': 'mesh_units',
            'help': ('Width of arrows of rays drawn when "--show-rays" '
                     'is passed.'),
        }),
        arguments.CompositeArgumentDescription(
            'traced_mesh_colormap', 'colormap',
            description=(
                'that should be used for query values '
                'if "--output-traced-mesh" is passed'
            ),
            defaults={
                'colormap': 'YlGn_r',
            },
        ),
    ]

    @classmethod
    def _write_output(cls, args, name, fname, output):
        r"""Write to an output file.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Name of the output to write.
            fname (str): Path of the file that should be written to.
            output (object): Output object to write to file.

        """
        if args.id == 'all' and name != 'traced_mesh':
            return
        super(RayTraceTask, cls)._write_output(args, name, fname, output)

    @property
    def verbose(self):
        r"""bool: Turn on log messages."""
        return self.args.verbose

    @readonly_cached_args_property
    def raytracer(self):
        r"""RayTracerBase: Ray tracer."""
        print("Re-creating ray tracer", self.args.time.value,
              self.args.age.value)
        assert self.geometryids is not None
        return get_class_registry().get(
            'raytracer', self.args.raytracer)(
                self.args, self.mesh,
                geometryids=self.geometryids)

    @cached_property
    def scene_area(self):
        r"""units.Quantity: Area that the scene covers."""
        if self.args.canopy != 'single':
            return self.args.plot_width * self.args.plot_length
        if self.args.virtual_canopy:
            return (self.args.virtual_plot_width
                    * self.args.virtual_plot_length)
        assert self.args.periodic_canopy
        return self.args.row_spacing * self.args.plant_spacing

    @cached_property
    def mesh(self):
        r"""ObjDict: Mesh that will be ray traced."""
        self.args.output_generate.assert_age_in_name(self.args)
        return self.get_output('generate')

    @cached_property
    def geometryids(self):
        return self.get_output('geometryids')[1]

    @cached_property
    def componentids(self):
        r"""np.ndarray: Component IDs for each face in the mesh."""
        return self.geometryids['componentids']

    @cached_property
    def plantids(self):
        r"""np.ndarray: Plant IDs for each face in the mesh."""
        return self.geometryids['plantids']

    @classmethod
    def adjust_args_internal(cls, args, skip=None, **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            skip (list, optional): Set of arguments to skip.
            **kwargs: Additional keyword arguments are passed to the
                parent class's method.

        """
        if skip is None:
            skip = []
        args.include_units = True
        if not GenerateTask.mesh_generated(args):
            args.periodic_canopy = False
            args.virtual_canopy = False
            args.canopy = 'external'
        LayoutTask.adjust_args_internal(
            args, skip=(skip + ['time', 'location']))
        for k in cls._virtual_argument_names:
            cls._arguments.getnested(f'virtual_{k}').adjust_args(
                args, default=getattr(args, k, None))
        if args.canopy == 'virtual':
            args.canopy = 'single'
        elif args.virtual_canopy != 'virtual':
            args.virtual_canopy = False
        super(RayTraceTask, cls).adjust_args_internal(
            args, skip=skip, **kwargs)
        if not cls.is_limits_base_class(args):
            args.dont_write_raytrace_limits = True

    @classmethod
    def _output_ext(cls, args, name, wildcards=None):
        r"""Determine the extension that should be used for output files.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Base name for variable to set.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.

        Returns:
            str: Output file extension.

        """
        if name == 'traced_mesh':
            return cls._outputs_external['generate']._output_ext(
                args, 'generate', wildcards=wildcards)
        return None

    @classmethod
    def extract_query(cls, query_values, query):
        r"""Extract a query value, scaling as necessary.

        Args:
            query_values (dict): Query values for each face.
            query (str): Query value to return.

        Returns:
            np.ndarray: Value for each face.

        """
        if query in query_values:
            return query_values[query]
        elif query == 'flux':
            return query_values['flux_density'] * query_values['areas']
        raise NotImplementedError(query)

    @classmethod
    def extract_query_average(cls, values, query):
        r"""Calculate average of total per plant.

        Args:
            values (dict): Query values for each face.
            query (str, list): Query value(s) to return total(s) for. If
                a list is provided, a dictionary mapping from query name
                to value will be returned.

        Returns:
            units.Quantity: Average for the specified query across plants.

        """
        def mean(x):
            if len(x) == 0:
                return 0.0
            if isinstance(x[0], units.Quantity):
                return units.Quantity(np.mean([xx.value for xx in x]),
                                      x[0].units)
            return np.mean(x)

        totals = cls.extract_query_total(values, query, per_plant=True)
        if isinstance(query, list):
            out = {k: mean([v[k] for v in totals.values()])
                   for k in query}
        else:
            out = mean(list(totals.values()))
        return out

    @classmethod
    def extract_query_averages(cls, values):
        r"""Calculate average of total per plant.

        Args:
            values (dict): Query values for each face.

        Returns:
            dict: Mapping between query names and averages.

        """
        query = [k for k in _query_options_phy if k in values]
        if (('flux' not in query
             and all(k in query for k in ['flux_density', 'areas']))):
            query.append('flux')
        return cls.extract_query_average(values, query)

    @classmethod
    def extract_query_total(cls, values, query, per_plant=False,
                            include_base=False):
        r"""Calculate sums over one query options.

        Args:
            values (dict): Query values for each face.
            query (str, list): Query value(s) to return total(s) for. If
                a list is provided, a dictionary mapping from query name
                to value will be returned.
            per_plant (bool, optional): If True, the query values should
                be totaled for each plant. If an int is provided,
                the number of unique plant IDs should match the provided
                value.
            include_base (bool, optional): If True, include the total
                for the whole scene when per_plant is True.

        Returns:
            units.Quantity: Total for the specified query.

        """
        def sum(x):
            if isinstance(x, units.QuantityArray):
                return units.Quantity(x.value.sum(), x.units)
            return x.sum()

        if 'flux' not in values and ((isinstance(query, list)
                                      and 'flux' in query)
                                     or query == 'flux'):
            values['flux'] = cls.extract_query(values, 'flux')

        if not per_plant:
            if isinstance(query, list):
                return {k: sum(values[k]) for k in query}
            return sum(values[query])
        out = {}
        plantids = values['plantids']
        plantids_unique = np.unique(plantids)
        if not isinstance(per_plant, bool):
            if ((len(plantids_unique) != per_plant
                 and len(plantids_unique) == 0)):
                if isinstance(query, list):
                    for k in query:
                        values[k] = np.zeros((per_plant, ))
                else:
                    values[query] = np.zeros((per_plant, ))
                plantids_unique = range(per_plant)
            else:
                assert len(plantids_unique) == per_plant
        for i in plantids_unique:
            idx = (plantids == i)
            if isinstance(query, list):
                out[str(i)] = {k: sum(values[k][idx]) for k in query}
            else:
                out[str(i)] = sum(values[query][idx])
        if include_base:
            if isinstance(query, list):
                out['total'] = {k: sum(values[k]) for k in query}
            else:
                out['total'] = sum(values[query])
        return out

    @classmethod
    def extract_query_totals(cls, values, per_plant=False):
        r"""Calculate sums over all query options.

        Args:
            values (dict): Query values for each face.
            per_plant (bool, optional): If True, the query values should
                also be totaled for each plant. If an int is provided,
                the number of unique plant IDs should match the provided
                value.

        Returns:
            dict: Mapping between query names and totals. If per_plant
                is True, this is nested as a value for each plant ID.

        """
        query = [k for k in _query_options_phy if k in values]
        if (('flux' not in query
             and all(k in query for k in ['flux_density', 'areas']))):
            query.append('flux')
        out = cls.extract_query_total(values, query, per_plant=per_plant)
        if per_plant:
            out['total'] = cls.extract_query_total(values, query)
        return out

    def calculate_query(self, query, prevent_output=False,
                        per_plant=False):
        r"""Calculate a query value for the scene.

        Args:
            query (str): Query value to return.
            prevent_output (bool, optional): If True, don't output
                raytracer results if they don't already exist.
            per_plant (bool, optional): If True, the query values should
                also be totaled for each plant. If an int is provided,
                the number of unique plant IDs should match the provided
                value. Only used for "total_*" query names.

        Returns:
            units.Quantity: Calculated query value.

        """
        query_sorted = {}

        def get_base(x):
            for prefix in _query_options_calc_prefix:
                if x.startswith(prefix):
                    out = x.split(prefix, 1)[-1]
                    query_sorted.setdefault(prefix, [])
                    query_sorted[prefix].append(out)
                    return out
            else:
                query_sorted.setdefault('', [])
                query_sorted[''].append(x)
                return x

        singular = (not isinstance(query, list))
        if singular:
            query = [query]
        query_base = set(get_base(k) for k in query)
        if (not prevent_output) or self.output_exists():
            values = self.get_output('raytrace')
        else:
            values.raytrace_scene(query=list(query_base))
        out = {}
        for prefix, kquery in query_sorted.items():
            if prefix == 'total_':
                iout = self.extract_query_total(values, kquery,
                                                per_plant=per_plant,
                                                include_base=True)
                if per_plant:
                    for k in kquery:
                        out.setdefault(f'{prefix}{k}', {})
                    for i in iout.keys():
                        for k in kquery:
                            out[f'{prefix}{k}'][i] = iout[i][k]
                    continue
            elif prefix == 'average_':
                iout = self.extract_query_average(values, kquery)
            elif prefix == 'scene_average_':
                iout = {
                    k: v / self.scene_area for k, v in
                    self.extract_query_total(values, kquery).items()
                }
            elif prefix == '':
                iout = {k: self.extract_query(values, k)
                        for k in kquery}
            else:
                raise NotImplementedError(prefix)
            for k, v in iout.items():
                out[f'{prefix}{k}'] = v
        if singular:
            return out[query[0]]
        return out

    @classmethod
    def calc_query_limits(cls, query_values, prev=None):
        r"""Compute limits from the query values.

        Args:
            query_values (dict): Query values for each face.

        Returns:
            dict: Limits on each query.

        """
        if isinstance(query_values, list):
            out = prev
            for x in query_values:
                out = cls.calc_query_limits(x, prev=out)
            return out
        out = {}
        for query in _query_options:
            out[query] = {}
            values = cls.extract_query(query_values, query)
            if isinstance(values, units.QuantityArray):
                values = values.value
            if len(values) == 0:
                out[query].update(
                    vmin_linear=np.nan,
                    vmin_log=np.nan,
                    vmax_linear=np.nan,
                    vmax_log=np.nan,
                )
            elif (values == 0).all():
                out[query].update(
                    vmin_linear=values[values >= 0].min(),
                    vmin_log=np.nan,
                    vmax_linear=values.max(),
                    vmax_log=np.nan,
                )
            else:
                out[query].update(
                    vmin_linear=values[values >= 0].min(),
                    vmin_log=values[values > 0].min(),
                    vmax_linear=values.max(),
                    vmax_log=values.max(),
                )
            if prev is not None:
                for k in ['vmin_linear', 'vmin_log']:
                    if ((out[query][k] == np.nan
                         or prev[query][k] < out[query][k])):
                        out[query][k] = prev[query][k]
                for k in ['vmax_linear', 'vmax_log']:
                    if ((out[query][k] == np.nan
                         or prev[query][k] > out[query][k])):
                        out[query][k] = prev[query][k]
        return out

    @classmethod
    def _adjust_color_limits(cls, self, cmap):
        r"""Set the minimum and maximum for color mapping.

        Args:
            self (TaskBase): Task that colormap limits are being set for.
            cmap (ColorMapArgument): Colormap argument information.

        """
        if cmap.limits_defined:
            return
        limits = self.get_output('raytrace_limits')
        cls.update_color_limits(self.args, cmap, limits=limits)

    @classmethod
    def update_color_limits(cls, args, cmap=None, limits=None,
                            name=None, force=False, **kwargs):
        r"""Set the minimum and maximum for color mapping.

        Args:
            args (argparse.Namespace): Parsed arguments.
            cmap (ColorMapArgument): Colormap argument information.
            limits (dict, optional): Set of calculated limits. If not
                provided, limits will be calculated via
                _generate_limits_class.
            name (str, optional): Name for limits to set if different
                than raytrace.
            force (bool, optional): If True, overwrite any existing
                arguments for the color limits.
            **kwargs: Additional keyword arguments are passed to
                _generate_limits_class if limits is not provided.

        Returns:
            bool: True if the limits were generated.

        """
        if name is None:
            name = cls._name
        if cmap is None:
            cmap = getattr(args, f'{name}_colormap')
        if (not force) and cmap.limits_defined:
            return False
        out = False
        if limits is None:
            limits = RayTraceTask._generate_limits_class(args, **kwargs)
            out = True
        vscaling = cmap.scaling
        limits = limits[args.query]
        if force or cmap.vmin is None:
            cmap.vmin = limits[f'vmin_{vscaling}']
        if force or cmap.vmax is None:
            cmap.vmax = limits[f'vmax_{vscaling}']
        cls.log_class(f'LIMITS[{cls._name}, {cmap.name}]: '
                      f'{cmap.vmin}, {cmap.vmax}')
        return out

    def _color_scene(self):
        r"""Color the geometry based on the raytracer results."""
        query_values = self.get_output('raytrace')
        if self.args.show_rays:
            inst = self.run_iteration(
                args_overwrite={'show_rays': False,
                                'output_traced_mesh': True},
            )
            mesh = inst.get_output('traced_mesh')
            self.raytracer = inst.raytracer
            mesh.append(
                generate_rays(inst.raytracer.ray_origins,
                              inst.raytracer.ray_directions,
                              ray_length=inst.raytracer.ray_lengths,
                              geom_format=type(mesh),
                              ray_color=self.args.ray_color.value,
                              ray_width=self.args.ray_width.value,
                              arrow_width=self.args.arrow_width.value)
            )
            return mesh
        cmap = self.args.traced_mesh_colormap
        self._adjust_color_limits(self, cmap)
        mesh = self.mesh
        face_values = self.extract_query(query_values, self.args.query)
        if isinstance(face_values, units.QuantityArray):
            face_values = np.array(face_values)
        vertex_values = self.raytracer.face2vertex(
            face_values, method='deposit')
        vertex_colors = utils.apply_color_map(
            vertex_values,
            color_map=cmap.colormap,
            vmin=cmap.vmin,
            vmax=cmap.vmax,
            scaling=cmap.scaling,
            highlight=self.args.highlight,
            highlight_color=self.args.highlight_color.value,
        )
        mesh.add_colors('vertex', vertex_colors)
        return mesh

    @classmethod
    def _check_for_plantids(cls, self, values):
        if not self.args.output_generate.generated:
            return True
        plantids = values['plantids']
        if len(plantids) == 0:
            return True
        plantids_unique = np.unique(plantids)
        if len(plantids_unique) != self.args.nrows * self.args.ncols:
            print(len(plantids), len(plantids_unique),
                  self.args.nrows * self.args.ncols)
            pdb.set_trace()
        return len(plantids_unique) == self.args.nrows * self.args.ncols

    def raytrace_scene(self, query=None):
        r"""Run the ray tracer on the selected geometry.

        Args:
            query (str, optional): Name of the field that should be
                calculated.

        Returns:
            dict: Dictionary of ray tracer queries.

        """
        if query is None:
            query = _query_options
        if isinstance(query, list):
            values = {}
            if 'flux' in query:
                query = query + [k for k in ['flux_density', 'areas']
                                 if k not in query]
            for k in query:
                if k == 'flux':
                    continue
                values[k] = self.raytrace_scene(query=k)
            if 'flux' in query:
                values['flux'] = self.extract_query(values, 'flux')
            return values
        if query == 'flux':
            values = self.raytrace_scene(query=['flux_density', 'areas'])
            return self.extract_query(values, query)
        query0 = self.args.query
        self.raytracer.args.query = query
        try:
            out = self.raytracer.raytrace()
        finally:
            self.raytracer.args.query = query0
        return out

    @classmethod
    def _generate_limits_class(cls, args, limits_args_base=None,
                               **kwargs):
        r"""Generate limits for the date in question.

        Args:
            args (argparse.Namespace): Parsed arguments.
            limits_args_base (dict): Arguments to use to generate base
                limits.
            **kwargs: Additional keyword arguments are passed to
                run_iteration_class.

        Returns:
            dict: Limits.

        """
        if limits_args_base is None:
            limits_args_base = cls.limits_args_class(args, base=True)
        args_overwrite = dict(
            limits_args_base,
            **{
                'query': None,
                'planting_date': None,
                'output_raytrace_limits': True,
                'dont_write_raytrace_limits': False,
            }
        )
        optional_output = cls._output_names(
            args, include_external=True, exclude_required=True)
        for k in optional_output:
            if k == 'raytrace_limits':
                continue
            args_overwrite[f'output_{k}'] = False
        print(80 * '-')
        print("GENERATING LIMITS FOR BASE")
        pprint.pprint(args_overwrite)
        out = cls.run_iteration_class(
            args, args_overwrite=args_overwrite,
            output_name='raytrace_limits',
            **kwargs
        )
        print("END GENERATING LIMITS FOR BASE")
        print(80 * '-')
        return out

    def _generate_limits(self):
        r"""Generate limits for the date in question.

        Returns:
            dict: Limits.

        """
        limits_args = self.limits_args_class(self.args)
        limits_args_base = self.limits_args_class(self.args, base=True)
        if limits_args != limits_args_base:
            return self._generate_limits_class(
                self.args, limits_args_base=limits_args_base,
                cached_outputs=self._cached_outputs,
                cache_outputs=['raytrace_limits'],
            )
        return self.calc_query_limits(self.get_output('raytrace'))

    @classmethod
    def is_limits_base_class(cls, args):
        r"""Check if the current arguments describe the set used to
        generate the base limits.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            bool: True if the current arguments are what is required
                for calculating limits.

        """
        limits_args = cls.limits_args_class(args)
        limits_args_base = cls.limits_args_class(args, base=True)
        return (limits_args == limits_args_base)

    @classmethod
    def limits_args_class(cls, args, base=False):
        r"""Get the arguments controling limits.

        Args:
            args (argparse.Namespace): Parsed arguments.
            base (bool, optional): If True, return the arguments that
                should be used to generate the limits rather than the
                current arguments.

        Returns:
            dict: Set of arguments controlling limits.

        """
        if base:
            out = {
                'time': 'noon',
                'date': None,
                'doy': 'summer_solstice',
            }
        else:
            out = {
                'time': args.time.solar_time_string,
                'date': None,
                'doy': args.time.solar_date_string,
            }
        if not GenerateTask.mesh_generated(args):
            return out
        if base:
            out.update(
                age='maturity',
                **GenerateTask.base_param_class(args)
            )
        else:
            out.update(
                id=args.id,
                data_year=args.data_year,
                age=args.time.crop_age_string,
            )
        return out

    def _merge_output(self, name, output, merged_param):
        r"""Merge the output for multiple sets of parameters values.

        Args:
            name (str): Name of the output to generate.
            output (dict): Mapping from tuples of parameter values to
               the output for the parameter values.
            merged_param (tuple): Names of the parameters that are being
                merged (the parameters specified by the tuple keys in
                output).

        Returns:
            object: Generated output.

        """
        if name != 'traced_mesh':
            return super(RayTraceTask, self)._merge_output(
                name, output, merged_param)
        return GenerateTask.merge_mesh(self.args, list(output.values()))

    def _generate_output(self, name):
        r"""Generate the specified output value.

        Args:
            name (str): Name of the output to generate.

        Returns:
            object: Generated output.

        """
        assert self.args.id != 'all'
        if name == 'raytrace':
            return self.raytrace_scene()
        elif name == 'raytrace_limits':
            return self._generate_limits()
        elif name == 'traced_mesh':
            return self._color_scene()
        super(RayTraceTask, self)._generate_output(name)


class RenderTask(TaskBase):
    r"""Class for rendering a 3D canopy."""

    _name = 'render'
    _output_info = {
        'render': {
            'ext': '.png',
            'base_output': 'raytrace',
            'description': 'the rendered image',
            'merge_all': True,
        },
        'render_camera': {
            'ext': '.json',
            'base_output': 'generate',
            'description': 'camera properties',
            'optional': True,
        },
    }
    _external_tasks = {
        RayTraceTask: {
            'exclude': [
                'show_rays', 'ray_color', 'ray_width', 'ray_length',
                'arrow_width', 'highlight', 'highlight_color',
                'traced_mesh_colormap',
            ],
            'modifications': {
                'query': {
                    'append_suffix_outputs': ['render'],
                },
                'virtual_canopy_properties': {
                    'append_suffix_outputs': ['render_camera'],
                },
            },
        },
    }
    _arguments = [
        (('--camera-direction', ), {
            'type': str,
            'help': ('Direction that camera should face. If not '
                     'provided, the camera will point to the center of '
                     'the scene from its location.'),
            'suffix_param': {},
        }),
        (('--camera-location', ), {
            'type': str,
            'help': ('Location of the camera. If not provided, one will '
                     'be determined that captures the entire scene from '
                     'the provided camera direction. If a direction is '
                     'also not provided, the camera will be centered '
                     'on the center of the scene facing down, '
                     'southeast at a distance that captures the entire '
                     'scene. If \"maturity\" is specified, the location '
                     'will be set for the mature plant (only valid for '
                     'generated meshes).'),
            'suffix_param': {'prefix': 'from_'},
        }),
        arguments.CompositeArgumentDescription(
            'scene_age', 'age',
            description=(
                'that the camera position should be calculated for '
                '(only valid for generated meshes)'
            ),
            ignore=['planting_date'],
            optional=True,
            suffix_param={}
        ),
        (('--camera-type', ), {
            'type': str,
            'choices': ['projection', 'orthographic'],  # 'spherical'],
            'default': 'projection',
            'help': ('Type of camera that should be used to render the '
                     'scene'),
            'suffix_param': {'noteq': 'projection'},
        }),
        arguments.DimensionArgumentDescription([
            (('--camera-fov-width', ), {
                'units': 'degrees', 'default': 45.0,
                'help': (
                    'Angular width of the camera\'s field of view (in '
                    'degrees) for a projection camera.'
                ),
                'suffix_param': {'conv': int},
            }),
            (('--camera-fov-height', ), {
                'units': 'degrees', 'default': 45.0,
                'help': (
                    'Angular height of the camera\'s field of view (in '
                    'degrees) for a projection camera.'
                ),
                'suffix_param': {'conv': int},
            }),
        ], name='camera_fov_dimensions', suffix_param={
            'cond': lambda x: (x.camera_type == 'projection'),
        }),
        (('--camera-up', ), {
            'type': str,
            'help': ('Up direction for the camera. If not provided, the '
                     'up direction for the scene will be assumed.'),
            'suffix_param': {'prefix': 'up', 'sep': 'x'},
        }),
        arguments.CompositeArgumentDescription(
            'background_color', 'color',
            description='that should be used for the scene',
            defaults={'color': 'transparent'},
            suffix_param={'noteq': 'transparent'},
        ),
        arguments.DimensionArgumentDescription([
            (('--image-nx', ), {
                'type': int,
                'help': (
                    'Number of pixels for the rendered image in the '
                    'horizontal direction. If not provided, but '
                    '--image-ny is provided, the value for --image-nx '
                    'will be determined from --image-ny by assuming '
                    'a constant resolution in both directions. If '
                    'neither are provided, --image-ny defaults to '
                    '1024.'
                ),
                'suffix_param': {},
            }),
            (('--image-ny', ), {
                'type': int,
                'help': (
                    'Number of pixels for the rendered image in the '
                    'vertical direction. If not provided, but '
                    '--image-nx is provided, the value for --image-ny '
                    'will be determined from --image-nx by assuming '
                    'a constant resolution in both directions. If '
                    'neither are provided, --image-ny defaults to '
                    '1024.'
                ),
                'suffix_param': {},
            }),
        ], name='resolution_dimensions', suffix_param={}),
        arguments.DimensionArgumentDescription([
            (('--image-width', ), {
                'units': 'cm',
                'units_arg': 'mesh_units',
                'help': (
                    'Width of the image (in cm). If not provided, '
                    'the width will be set based on the camera '
                    'position and type such that the entire scene '
                    'is captured.'
                ),
                'suffix_param': {'conv': int},
            }),
            (('--image-height', ), {
                'units': 'cm',
                'units_arg': 'mesh_units',
                'help': (
                    'Height of the image (in cm). If not provided, '
                    'the height will be set based on the camera '
                    'position and type such that the entire scene '
                    'is captured.'
                ),
                'suffix_param': {'conv': int},
            }),
        ], name='image_dimensions', suffix_param={
            'suffix': 'cm',
        }),
        (('--resolution', ), {
            'units': 'cm**-1',  # 'default': 5,
            'help': ('Resolution that the scene should be rendered with '
                     'in pixels per centimeter. If provided, any '
                     'values provided for --image-nx and --image-ny '
                     'will be ignored. If not provided, the resolution '
                     'in each direction will be determined by '
                     '--image-nx and --image-ny.'),
            'suffix_param': {
                'conv': int,
                'suffix': 'percm',
            },
        }),
        (('--scene-mins', ), {
            'help': 'Minimum extent of scene along each dimension',
            'suffix_param': {
                'sep': 'x',
                'prefix': 'SCMIN',
            },
        }),
        (('--scene-maxs', ), {
            'help': 'Maximum extent of scene along each dimension',
            'suffix_param': {
                'sep': 'x',
                'prefix': 'SCMAX',
            },
        }),
        arguments.CompositeArgumentDescription(
            'render_colormap', 'colormap',
            description=(
                'that should be used for query values '
                'if "--output-render" is passed'
            ),
            defaults={
                'colormap': 'viridis',
                'colormap_scaling': 'log',
            },
        ),
    ]

    @classmethod
    def adjust_args_internal(cls, args, **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to the
                parent class's method.

        """
        args.show_rays = False
        if ((args.camera_direction is None
             and args.camera_location is None)):
            args.camera_direction = 'downnortheast'
        super(RenderTask, cls).adjust_args_internal(args, **kwargs)
        if not cls.is_camera_base_class(args):
            args.dont_write_render_camera = True

    @classmethod
    def adjust_args_step(cls, args, vary):
        r"""Adjust the parsed arguments before the class is run in a
        temporal iteration.

        Args:
            args (argparse.Namespace): Parsed arguments.
            vary (str): Name of argument being varied.

        """
        if vary == 'time':
            if args.start_time.age == args.stop_time.age:
                return
            args.scene_age = args.stop_age

    @property
    def raytracer(self):
        r"""RayTracerBase: Ray tracer."""
        return self.external_tasks['raytrace'].raytracer

    def _render_scene(self):
        r"""Render the scene using a ray tracer.

        Returns:
            np.ndarray: Pixel color data.

        """
        query_values = self.get_output('raytrace')
        cmap = self.args.render_colormap
        RayTraceTask._adjust_color_limits(self, cmap)
        face_values = RayTraceTask.extract_query(
            query_values, self.args.query)
        if isinstance(face_values, units.QuantityArray):
            face_values = np.array(face_values)
        pixel_values = self.raytracer.render(face_values)
        if isinstance(pixel_values, units.QuantityArray):
            pixel_values = np.array(pixel_values)
        # self.set_output('pixel_values', pixel_values)
        pixel_values = (pixel_values.T)[::-1, :]
        image = utils.apply_color_map(
            pixel_values,
            color_map=cmap.colormap,
            vmin=cmap.vmin,
            vmax=cmap.vmax,
            scaling=cmap.scaling,
            highlight=(pixel_values < 0),
            highlight_color=self.args.background_color.value,
            include_alpha=(len(self.args.background_color.value) == 4)
        )
        return image

    @classmethod
    def is_camera_base_class(cls, args):
        r"""Check if the current arguments describe the set used to
        generate the base camera.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            bool: True if the current arguments are what is required
                for setting up the base camera.

        """
        camera_args = cls.camera_args_class(args)
        camera_args_base = cls.camera_args_class(args, base=True)
        return (camera_args == camera_args_base)

    @classmethod
    def camera_args_class(cls, args, base=False):
        r"""Get the arguments controling the camera.

        Args:
            args (argparse.Namespace): Parsed arguments.
            base (bool, optional): If True, return the arguments that
                should be used to generate the camera rather than the
                current arguments.

        Returns:
            dict: Set of arguments controlling the camera.

        """
        if not GenerateTask.mesh_generated(args):
            return {}
        if base:
            out = {
                'age': 'maturity',
                'scene_age': None,
            }
            if args.scene_age.args['age']:
                out['age'] = args.scene_age.args['age']
        else:
            out = {
                'age': args.time.args['age'],
                'scene_age': args.scene_age.args['age'],
            }
        if args.virtual_canopy:
            out['canopy'] = 'virtual'
            out['nrows'] = args.virtual_nrows
            out['ncols'] = args.virtual_ncols
        else:
            out['canopy'] = args.canopy
            out['nrows'] = args.nrows
            out['ncols'] = args.ncols
        return out

    @classmethod
    def _generate_camera_class(cls, args, camera_args_base=None,
                               **kwargs):
        r"""Generate camera properties for a specific scene age.

        Args:
            args (argparse.Namespace): Parsed arguments.
            camera_args_base (dict): Arguments to use to generate base
                camera properties.
            **kwargs: Additional keyword arguments are passed to
                run_iteration_class.

        Returns:
            dict: Camera properties.

        """
        if camera_args_base is None:
            camera_args_base = cls.camera_args_class(args, base=True)
        args_overwrite = dict(
            camera_args_base,
            **{
                'output_render': False,
                'output_render_camera': True,
                'dont_write_render_camera': False,
                'scene_mins': None,
                'scene_maxs': None,
                'time': None,
            }
        )
        optional_output = cls._output_names(
            args, include_external=True, exclude_required=True)
        for k in optional_output:
            if k == 'render_camera':
                continue
            args_overwrite[f'output_{k}'] = False
        print(80 * '-')
        print("GENERATING CAMERA PROPERTIES FOR BASE")
        pprint.pprint(args_overwrite)
        out = cls.run_iteration_class(
            args, args_overwrite=args_overwrite,
            output_name='render_camera',
            **kwargs
        )
        print("END GENERATING CAMERA PROPERTIES FOR BASE")
        print(80 * '-')
        return out

    def _generate_camera(self):
        r"""Generate camera properties for a specific scene age.

        Returns:
            dict: Camera properties.

        """
        camera_args = self.camera_args_class(self.args)
        camera_args_base = self.camera_args_class(self.args, base=True)
        if camera_args != camera_args_base:
            return self._generate_camera_class(
                self.args, camera_args_base=camera_args_base,
                cached_outputs=self._cached_outputs,
                cache_outputs=['render_camera'],
            )
        out = {}
        for k in ['camera_type', 'camera_fov_width',
                  'camera_fov_height']:
            out[k] = getattr(self.args, k)
        for k in ['camera_direction', 'camera_up', 'camera_location',
                  'image_nx', 'image_ny',
                  'image_width', 'image_height',
                  'resolution']:
            out[k] = getattr(self.raytracer, k)
        for k in ['mins', 'maxs']:
            out[f'scene_{k}'] = getattr(
                self.raytracer.virtual_scene_model, k)
        return out

    def _merge_output(self, name, output, merged_param):
        r"""Merge the output for multiple sets of parameters values.

        Args:
            name (str): Name of the output to generate.
            output (dict): Mapping from tuples of parameter values to
               the output for the parameter values.
            merged_param (tuple): Names of the parameters that are being
                merged (the parameters specified by the tuple keys in
                output).

        Returns:
            object: Generated output.

        """
        if name == 'render':
            return np.concatenate(list(output.values()), axis=1)
        return super(RenderTask, self)._merge_output(
            name, output, merged_param)

    def _generate_output(self, name):
        r"""Generate the specified output value.

        Args:
            name (str): Name of the output to generate.

        Returns:
            object: Generated output.

        """
        assert self.args.id != 'all'
        if name == 'render_camera':
            return self._generate_camera()
        elif name == 'render':
            if self.args.scene_age.age is not None:
                before = self.args.output_render.path
                camera_args = self.get_output('render_camera')
                for k, v in camera_args.items():
                    setattr(self.args, k, v)
                after = self.args.output_render.path
                assert after == before
            return self._render_scene()
        return super(RenderTask, self)._generate_output(name)


class TotalsTask(TemporalTaskBase):
    r"""Class for plotting the flux on a geometry as a function of time."""
    _name = 'totals'
    _step_task = RayTraceTask
    _output_info = {
        'totals': {
            'base_prefix': True,
            'ext': '.json',
            'description': 'raytraced query totals',
        },
        'totals_plot': {
            'ext': '.png',
            'base_output': 'totals',
            'description': 'a plot of raytraced query totals',
            'optional': True,
            'merge_all': True,
            'merge_all_output': 'totals',
        },
    }
    _external_tasks = {
        RayTraceTask: {
            'exclude': [
                'show_rays', 'ray_color', 'ray_width', 'ray_length',
                'arrow_width', 'highlight', 'highlight_color',
                'traced_mesh_colormap',
            ],
            'modifications': {
                'query': {
                    'default': 'total_flux',
                    'choices': _query_options_calc,
                    'append_suffix_skip_outputs': ['totals'],
                    'append_suffix_outputs': ['totals_plot'],
                },
            },
        },
    }
    _arguments = [
        (('--per-plant', ), {
            'action': 'store_true',
            'help': ('Plot the totals on a per-plant basis (valid for '
                     'totals_plot only'),
        }),
    ]

    @classmethod
    def _read_output(cls, args, name, fname):
        r"""Load an output file produced by this task.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Name of the output to read.
            fname (str): Path of file that should be read from.

        Returns:
            object: Contents of the output file.

        """
        out = super(TotalsTask, cls)._read_output(args, name, fname)
        if name == 'totals':
            out['times'] = [
                datetime.fromisoformat(x) for x in out['times']
            ]
        return out

    @classmethod
    def _write_output(cls, args, name, fname, output):
        r"""Write to an output file.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Name of the output to write.
            fname (str): Path of the file that should be written to.
            output (object): Output object to write to file.

        """
        if name == 'totals':
            output = dict(output,
                          times=[x.isoformat() for x in output['times']])
        super(TotalsTask, cls)._write_output(args, name, fname, output)

    @classmethod
    def adjust_args_internal(cls, args, **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to the
                parent class's method.

        """
        if args.canopy in ['single', 'virtual']:
            args.per_plant = False
        super(TotalsTask, cls).adjust_args_internal(args, **kwargs)

    @cached_property
    def time_marker(self):
        r"""matplotlib.lines.line2D: Vertical line marking a time."""
        return self.axes.axvline(x=self.start_time, color=(1, 1, 1),
                                 alpha=0.5, linewidth=10)

    def mark_time(self, time):
        r"""Mark a time with a vertical line on the figure.

        Args:
            time (datetime.datetime): Time that should be marked.

        """
        self.time_marker.set_xdata([time, time])

    def plot_query(self, query, times, values, id=None, plantid=None,
                   reset=False):
        r"""Plot the data for a single crop ID.

        Args:
            query (str): Name of property contained by values.
            times (np.ndarray): Times.
            values (np.ndarray): Query values for each time step.
            id (str, optional): ID of the crop that should be used for
                the label.
            plantid (int, optional): Plant ID for provided data.
            reset (bool, optional): Reset the figure.

        """
        ax = self.axes
        first = (reset or not hasattr(self, '_lines'))
        if first:
            self._lines = {}
            ax.cla()
        if id is None:
            id = self.args.id
        iclass = len(self._lines)
        self._lines.setdefault(id, {'interior': 0, 'exterior': 0})
        linestyles = ['-', ':', '--', '-.']
        colors = ['blue', 'orange', 'purple', 'teal']
        if first:
            ylabel = query.title()
            if isinstance(values, units.QuantityArray):
                ylabel += f" ({values.units})"
            elif isinstance(values[0], units.Quantity):
                ylabel += f" ({values[0].units})"
            ax.set_xlabel('Time')
            ax.set_ylabel(ylabel)
        layout_task = self.output_task('layout')
        loc = 0
        locStr = None
        label = None
        if plantid is not None:
            if layout_task.isExteriorPlant(int(plantid)):
                loc = 1
                locStr = 'exterior'
            else:
                loc = 0
                locStr = 'interior'
            self._lines[id] += 1
            if self._lines[locStr] == 1:
                label = f'{locStr.title()} plants ({id})'
        else:
            label = id
        color = colors[iclass]
        style = linestyles[loc]
        ax.plot(times, values, label=label, color=color,
                linestyle=style)

    def plot_data(self, totals, id=None, reset=False):
        r"""Plot the data for a single crop ID.

        Args:
            totals (dict): Mapping between query name and dictionaries
                of totals for each component.
            id (str, optional): ID of the crop that should be used for
                the label.
            reset (bool, optional): Reset the figure.

        """
        if id is None:
            id = self.args.id
        times = totals['times']
        totals = totals[self.args.query]
        if self.args.query.startswith('total_'):
            if self.args.per_plant:
                for i, v in totals.items():
                    if i == 'total':
                        continue
                    self.plot_query(self.args.query, times, v,
                                    id=id, plantid=i, reset=reset)
                    reset = False
                return
            totals = totals['total']
        self.plot_query(self.args.query, times, totals, id=id,
                        reset=reset)

    def finalize_step(self, x):
        r"""Finalize the output from a step.

        Args:
            x (object): Result of step.

        Returns:
            object: Finalized step result.

        """
        per_plant = (self.args.nrows * self.args.ncols)
        query = _query_options_calc
        value = x.calculate_query(query, per_plant=per_plant)
        return (x.args.time.time, value)

    def join_steps(self, xlist):
        r"""Join the output form all of the steps.

        Args:
            xlist (list): Result of all steps.

        Returns:
            object: Joined output from all steps.

        """
        assert self.args.id != 'all'
        times = [x[0] for x in xlist]
        values = [x[1] for x in xlist]
        N = len(values)
        out = {}
        for k, v in values[0].items():
            if k.startswith('total_'):
                out[k] = {i: np.zeros(N, dtype='float64')
                          for i in v.keys()}
            else:
                out[k] = np.zeros(N, dtype='float64')
        field_units = {}

        def get_value(k, x):
            if not isinstance(x, units.Quantity):
                return x
            if k not in field_units:
                field_units[k] = str(x.units)
            return float(x.value)

        for idx, value in enumerate(values):
            for k, v in value.items():
                if k.startswith('total_'):
                    for i, x in v.items():
                        out[k][i][idx] = get_value(k, x)
                else:
                    out[k][idx] = get_value(k, v)
        for k, unit in field_units.items():
            if k.startswith('total_'):
                for i in list(out[k].keys()):
                    out[k][i] = parse_quantity(out[k][i], unit)
            else:
                out[k] = parse_quantity(out[k], unit)
        out['times'] = times
        return out

    def _merge_output(self, name, output, merged_param):
        r"""Merge the output for multiple sets of parameters values.

        Args:
            name (str): Name of the output to generate.
            output (dict): Mapping from tuples of parameter values to
               the output for the parameter values.
            merged_param (tuple): Names of the parameters that are being
                merged (the parameters specified by the tuple keys in
                output).

        Returns:
            object: Generated output.

        """
        if name == 'totals_plot':
            for param, values in output.items():
                idstr = ', '.join([str(x) for x in param])
                # kws = {k: v for k, v in zip(merged_param, param)}
                self.plot_data(values, id=idstr)
            self.axes.legend()
            return self.figure
        return super(TotalsTask, self)._merge_output(
            name, output, merged_param)

    def _generate_output(self, name):
        r"""Generate the specified output value.

        Args:
            name (str): Name of the output to generate.

        Returns:
            object: Generated output.

        """
        if name == 'totals_plot':
            assert self.args.id != 'all'
            out = self.get_output('totals')
            self.plot_data(out)
            self.axes.legend()
            return self.figure
        return super(TotalsTask, self)._generate_output(name)


class AnimateTask(TemporalTaskBase):
    r"""Class for producing an animation."""

    _name = 'animate'
    _step_task = RenderTask
    _step_args_preserve = ['render_colormap']
    _output_info = {
        'animate': {
            'base_output': 'render',
            'description': 'the animation',
        },
    }
    _external_tasks = {
        RenderTask: {
            'modifications': {
                'output_render': {
                    'default': True,
                },
            }
        },
        TotalsTask: {
            'optional': True,
            'suffix_index': 1,
        },
    }
    _arguments = [
        (('--movie-format', ), {
            'type': str, 'choices': ['mp4', 'mpeg', 'gif'],
            'default': 'gif',
            'help': 'Format that the movie should be output in',
            'suffix_param': {'prefix': '.', 'cond': False},
        }),
        (('--frame-rate', ), {
            'type': int, 'default': 1,
            'help': ('The frame rate that should be used for the '
                     'generated movie in frames per second'),
            'suffix_param': {'suffix': 'fps', 'noteq': 1},
        }),
        (('--inset-totals', ), {
            'action': 'store_true',
            'help': ('Inset a plot of the query total below the '
                     'image.'),
            'suffix_param': {'value': 'totals'},
        }),
    ]

    def __init__(self, *args, **kwargs):
        self._inset_figure = None
        super(AnimateTask, self).__init__(*args, **kwargs)

    @classmethod
    def _write_output(cls, args, name, fname, output):
        r"""Write to an output file.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Name of the output to write.
            fname (str): Path of the file that should be written to.
            output (object): Output object to write to file.

        """
        utils.write_movie(output, fname, frame_rate=args.frame_rate,
                          verbose=args.verbose)

    @classmethod
    def _output_ext(cls, args, name, wildcards=None):
        r"""Determine the extension that should be used for output files.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Base name for variable to set.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.

        Returns:
            str: Output file extension.

        """
        return cls._arguments['movie_format'].generate_suffix(
            args, name, wildcards=wildcards, force=True,
        )

    @property
    def totals_task(self):
        r"""TotalsTask: Totals instance."""
        return self.external_tasks['totals']

    def finalize_step(self, x):
        r"""Finalize the output from a step.

        Args:
            x (object): Result of step.

        Returns:
            object: Finalized step result.

        """
        assert isinstance(x.output_file('render'), str)
        if not self.args.inset_totals:
            return x.output_file('render')
        old_frame = x.output_file('render')
        new_frame = '_totals'.join(os.path.splitext(old_frame))
        if os.path.isfile(new_frame) and not self.args.overwrite_render:
            return new_frame
        old_data = x.get_output('render')
        if self._inset_figure is None:
            self._inset_figure = self.totals_task.figure
            print('OLD_IMAGE', old_data.shape)
            pdb.set_trace()
            width_px = old_data.shape[0]
            height_px = int(0.2 * width_px)
            dpi = self._inset_figure.get_dpi()
            self._inset_figure.set_size_inches(width_px / dpi,
                                               height_px / dpi)
        self.totals_task.mark_time(x.args.time.time)
        add_data = self.totals_task.raw_figure_data
        print('NEW_IMAGE', add_data.shape)
        pdb.set_trace()
        new_data = np.concatenate([old_data, add_data])
        print('CONCAT', new_data)
        utils.write_png(new_data, new_frame, verbose=self.args.verbose)
        return new_frame


class MatchQuery(OptimizationTaskBase):
    r"""Class for matching raytrace query."""

    _step_task = RayTraceTask
    _name = 'match_query'
    _final_outputs = ['totals_plot']
    _arguments = [
        (('--goal-id', ), {
            'help': (
                'ID that the goal of the optimization should be matched '
                'to.'),
            'suffix_param': {'prefix': 'matchTo_'},
        }),
        (('--goal', ), {
            'help': 'Goal of the optimization.',
            'default': 'scene_average_flux',
            'choices': _query_options,
            'suffix_param': {},
        }),
        (('--vary', ), {
            'type': str,
            'help': 'Argument that should be varied.',
            'default': 'row_spacing',
            'suffix_param': {'prefix': 'vs_'},
        }),
        OptimizationTaskBase._arguments['method'],
    ]

    @classmethod
    def adjust_args_internal(cls, args, **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to the
                parent class's method.

        """
        args.final_args = {}
        for k in ['canopy', 'periodic_canopy', 'nrows', 'ncols',
                  'periodic_canopy_count', 'plot_width', 'plot_length']:
            args.final_args[k] = getattr(args, k)
        if args.canopy not in ['single', 'virtual']:
            # Regenerating unique canopies is very time intensive
            args.canopy = 'virtual'
        if args.goal_id is None:
            ids = GenerateTask.all_ids_class(args)
            assert ids
            for x in ids:
                if x != args.id:
                    args.goal_id = x
                    break
        super(MatchQuery, cls).adjust_args_internal(args, **kwargs)

    @cached_property
    def goal(self):
        r"""units.Quantity: Value that should be achieved."""
        assert self.args.id != self.args.goal_id
        args_overwrite = {
            'id': self.args.goal_id,
            self.args.vary: getattr(self.args, self.args.vary)
        }
        inst = self.run_iteration(
            cls=self._step_task,
            args_overwrite=args_overwrite,
        )
        out = self.finalize_step(inst, force_output=True)
        self.log(f'GOAL: {out}', force=True)
        return out

    def finalize_step(self, x, force_output=False):
        r"""Finalize the output from a step.

        Args:
            x (object): Result of step.
            force_output (bool, optional): If True, force output to
                disk.

        Returns:
            object: Finalized step result.

        """
        if getattr(self, '_prev_instance', None) is None:
            force_output = True
        return x.calculate_query(
            self.args.goal, prevent_output=(not force_output)
        )

    def final_output_args(self, name):
        r"""Get the arguments that should be used generate the final
        output.

        Args:
            name (str): Name of the final output to generate.

        Returns:
            dict: Arguments to use.

        """
        out = copy.deepcopy(self.args.final_args)
        if name == 'totals_plot':
            out['query'] = self.args.goal
            if ((out['canopy'] not in ['single', 'virtual']
                 and out['query'].startswith('total_'))):
                out['per_plant'] = True
        return out

    def _generate_output(self, name):
        r"""Generate the specified output value.

        Args:
            name (str): Name of the output to generate.

        Returns:
            object: Generated output.

        """
        out = super(MatchQuery, self)._generate_output(name)
        if name == 'totals_plot':
            args_overwrite = dict(
                self.final_output_args(name),
                id=self.args.goal_id,
            )
            goal_totals = self.run_iteration(
                cls=TotalsTask,
                args_overwrite=args_overwrite,
                output_name='totals',
            )
            out = super(MatchQuery, self)._generate_output(
                name, output_name='instance')
            match_totals = out.get_output('totals')
            out.plot_data(goal_totals, id=self.args.goal_id, reset=True)
            out.plot_data(match_totals, id=self.args.id)
            out.axes.legend()
            return out.figure
        return super(MatchQuery, self)._generate_output(name)
