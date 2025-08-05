import os
import pdb
import numpy as np
import itertools
import pprint
from datetime import datetime
from collections import OrderedDict
from yggdrasil import units
from yggdrasil.serialize.PlySerialize import PlyDict
from yggdrasil.serialize.ObjSerialize import ObjDict
from canopy_factory import utils
from canopy_factory.utils import (
    rapidjson, RegisteredClassBase, get_class_registry,
    parse_quantity, parse_axis,
    cached_property, cached_args_property, readonly_cached_args_property,
    # Geometry
    scene2geom, _lpy_rays,
)
from canopy_factory.cli import (
    TaskBase, TemporalTaskBase,
    RepeatIterationError, OutputArgument,
    OptimizationTaskBase,
)
from canopy_factory.crops import GenerateTask


_query_options = [
    'flux_density', 'flux', 'hits', 'areas', 'plantids', 'componentids',
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
    lsys = Lsystem(_lpy_rays, param)
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
        # print(f'{np.logical_not(self.area_mask).sum()} '
        #       f'faces have areas of 0')
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
    def scene_mins(self):
        r"""np.ndarray: Minimum scene vertices in each dimension."""
        if getattr(self.args, 'scene_mins', None) is not None:
            return self.args.scene_mins
        if self.mesh_dict['vertex'].shape[0] == 0:
            return np.zeros((3, ), dtype=self.mesh_dict['vertex'].dtype)
        return self.mesh_dict['vertex'].min(axis=0)

    @cached_args_property
    def scene_maxs(self):
        r"""np.ndarray: Maximum scene vertices in each dimension."""
        if getattr(self.args, 'scene_maxs', None) is not None:
            return self.args.scene_maxs
        if self.mesh_dict['vertex'].shape[0] == 0:
            return np.zeros((3, ), dtype=self.mesh_dict['vertex'].dtype)
        return self.mesh_dict['vertex'].max(axis=0)

    @cached_args_property
    def scene_limits(self):
        r"""np.ndarray: Corners of a box containing the scene."""
        limits = np.vstack([self.scene_mins, self.scene_maxs])
        xx, yy, zz = np.meshgrid(limits[:, 0], limits[:, 1], limits[:, 2])
        out = np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T
        return out

    @cached_args_property
    def scene_center(self):
        r"""np.ndarray: Coordinates of the scene's center."""
        return (self.scene_maxs + self.scene_mins) / 2

    @cached_args_property
    def scene_dim(self):
        r"""np.ndarray: Scene's dimensions in each direction."""
        return (self.scene_maxs - self.scene_mins)

    @cached_args_property
    def ground(self):
        r"""np.ndarray: """
        return (
            np.dot(self.scene_dim / 2, self.north) * self.north
            + np.dot(self.scene_dim / 2, self.east) * self.east
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
            out = self.scene_center - self.camera_location
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
            self.scene_limits - self.scene_center,
            self.camera_right)))
        camera_distance = np.abs(
            fov_width / np.tan(self.args.camera_fov_width / 2.0))
        if camera_distance < self.clipping_distance.value:
            camera_distance = self.clipping_distance.value
        out = (
            self.scene_center
            - (camera_distance * self.camera_direction)
        )
        if isinstance(out, units.QuantityArray) and out.is_dimensionless():
            out = np.array(out)
        return out

    @cached_args_property
    def camera_distance(self):
        r"""float: Distance between the camera and the scene scenter."""
        return units.Quantity(
            np.linalg.norm(self.scene_center - self.camera_location),
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
                self.scene_limits - self.camera_location,
                self.camera_right))),
            2 * np.max(np.abs(np.dot(
                self.scene_limits - self.camera_location,
                self.camera_up))),
            2 * np.max(np.abs(np.dot(
                self.scene_limits - self.camera_location,
                self.camera_direction))),
        ])

    @cached_args_property
    def clipping_distance(self):
        r"""float: Maximum distance of any scene limits from the scene
        center along the camera line-of-sight."""
        return units.Quantity(
            np.max(np.abs(np.dot(
                self.scene_limits - self.scene_center,
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
        # if face_scalar.shape == self.idx_faces.shape:
        #     face_scalar = face_scalar[self.idx_faces]
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
        if self.args.periodic_canopy == 'scene':
            from hothouse.scene import PeriodicScene as Scene
            kws.update(period=self.args.periodic_period.astype('f4'),
                       direction=self.args.periodic_direction.astype('f4'),
                       count=(self.args.periodic_canopy_count
                              * np.ones((3, ), 'i4')))
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
    def solar_blaster(self):
        r"""hothouse.blaster.SolarBlaster: Blaster for sun."""
        # TODO: Add units to parser
        self.log(f"Total PPFD"
                 f"\n   direct = {self.solar_model.ppfd_direct}"
                 f"\n   diffuse = {self.solar_model.ppfd_diffuse}",
                 force=True)
        kws = {}
        if self.args.periodic_canopy == 'rays':
            kws.update(
                period=self.args.periodic_period.astype('f4'),
                periodic_direction=self.args.periodic_direction.astype('f4'),
                periodic_count=(self.args.periodic_canopy_count
                                * np.ones((3, ), 'i4')),
            )
        return self.scene.get_sun_blaster(
            self.solar_model.latitude.value,
            self.solar_model.longitude.value,
            self.solar_model.time,
            direct_ppfd=self.solar_model.ppfd_direct,
            diffuse_ppfd=self.solar_model.ppfd_diffuse,
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
                 f'({self.args.age.value}) with sun '
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
            'base': 'generate',
            'ext': '.csv',
            'description': (
                'the query values for each face in the mesh'
            ),
        },
        'raytrace_limits': {
            'base': 'raytrace',
            'ext': '.json',
            'description': 'limits on raytrace query values',
            'optional': True,
        },
        'traced_mesh': {
            'base': 'raytrace',
            'description': (
                'the mesh with faces colored by a ray tracer result'
            ),
            'optional': True,
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
                },
            }
        },
    }
    _composite_arguments = {
        'time': {
            'time': {
                'description': ' that the sun should be modeled for',
                'defaults': {
                    'hour': 'noon',
                    'date': '2024-06-21',
                },
            },
        },
        'color': {
            'ray_color': {
                'description': (
                    ' that should be used for rays when "--show-rays" '
                    'is passed'
                ),
                'defaults': {
                    'color': 'red',
                },
            },
            'highlight_color': {
                'description': (
                    'that should be used for highlighted faces if '
                    '"--highlight" is passed'
                ),
                'defaults': {
                    'color': 'magenta',
                },
            },
        },
    }
    _argument_conversions = {
        'mesh_units': [
            'ground_height',
            'ray_width', 'ray_length', 'arrow_width',
        ],
    }
    _arguments = [
        (('--raytracer', ), {
            'type': str, 'default': 'hothouse',
            'choices': list(get_class_registry().keys('raytracer')),
            'help': 'Name of the ray tracer that should be used.',
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
        }),
        (('--any-direction', ), {
            'action': 'store_true',
            'help': ('Allow light to be deposited by the ray tracer '
                     'from any direction relative to the surface. If '
                     'not set, only ray intercepting a surface from '
                     'the \"top\" will be counted.'),
        }),
        (('--multibounce', ), {
            'action': 'store_true',
            'help': ('Include multiple bounces when performing the '
                     'trace.'),
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
        }),
        # (('--query-units', ), {
        #     'type': str,
        #     'help': ('Units that query should be expressed in. Defaults '
        #              'to Watts for query=\"flux\" and unitless '
        #              'otherwise.'),
        # }),
        (('--show-rays', ), {
            'action': "store_true",
            'help': ('Show the rays in the generated mesh if '
                     '"--output-traced-mesh" is passed.'),
        }),
        (('--highlight', ), {
            'type': str, 'choices': ['min', 'max'],
            'help': ('Highlight the face with the \"min\" or \"max\" '
                     'query value in the resulting (only valid if '
                     '"--output-traced-mesh" is passed).'),
        }),
        (('--ray-width', ), {
            'type': parse_quantity, 'default': 1.0, 'units': 'cm',
            'help': 'Width of rays drawn when "--show-rays" is passed.',
        }),
        (('--ray-length', ), {
            'type': parse_quantity, 'default': 10.0, 'units': 'cm',
            'help': ('Length of rays drawn when "--show-rays" is '
                     'passed. A negative value will cause the distance '
                     'to the scene to be used for the ray length.'),
        }),
        (('--arrow-width', ), {
            'type': parse_quantity, 'default': 2.0, 'units': 'cm',
            'help': ('Width of arrows of rays drawn when "--show-rays" '
                     'is passed.'),
        }),
        (('--colormap', ), {
            'type': str, 'default': 'YlGn_r',
            'help': ('Name of the matplotlib color map that should be '
                     'used to map query values for each face to colors '
                     'if "--output-traced-mesh" is passed'),
            'subparser_specific_dest': True,
        }),
        (('--color-vmin', ), {
            'type': parse_quantity,
            'help': ('Query value that should be mapped to the lowest '
                     'value for the colormap if "--output-traced-mesh" '
                     'is passed'),
            'subparser_specific_dest': True,
        }),
        (('--color-vmax', ), {
            'type': parse_quantity,
            'help': ('Query value that should be mapped to the highest '
                     'value for the colormap if "--output-traced-mesh" '
                     'is passed'),
            'subparser_specific_dest': True,
        }),
        (('--colormap-scaling', ), {
            'type': str, 'choices': ['linear', 'log'],
            'default': 'linear',
            'help': ('Scaling that should be used to map query values '
                     'to colors in the color map if '
                     '"--output-traced-mesh" is passed'),
            'subparser_specific_dest': True,
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
        if name == 'traced_mesh':
            return utils.read_3D(fname, file_format=args.mesh_format,
                                 verbose=args.verbose,
                                 include_units=args.include_units)
        elif name == 'raytrace_limits':
            with open(fname, 'r') as fd:
                return rapidjson.load(fd)
        elif name == 'raytrace':
            return utils.read_csv(fname, verbose=args.verbose)
        return super(RayTraceTask, cls)._read_output(args, name, fname)

    @classmethod
    def _write_output(cls, args, name, fname, output):
        r"""Write to an output file.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Name of the output to write.
            fname (str): Path of the file that should be written to.
            output (object): Output object to write to file.

        """
        if args.id == 'all_split' and name != 'traced_mesh':
            return
        if name == 'traced_mesh':
            return utils.write_3D(output, fname,
                                  file_format=args.mesh_format,
                                  verbose=args.verbose)
        elif name == 'raytrace_limits':
            assert output
            with open(fname, 'w') as fd:
                rapidjson.dump(output, fd, write_mode=rapidjson.WM_PRETTY)
            return
        elif name == 'raytrace':
            utils.write_csv(output, fname, verbose=args.verbose)
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
    def adjust_args_internal(cls, args):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        args.include_units = True
        if ((isinstance(args.output_generate, str)
             or (isinstance(args.output_generate, OutputArgument)
                 and args.output_generate.generated))):
            args.periodic_canopy = False
        super(RayTraceTask, cls).adjust_args_internal(args)
        assert args.output_generate.generated

    def reset_outputs(self, **kwargs):
        r"""Remove existing files that should be overwritten.

        Args:
            **kwargs: Keyword arguments are passed to base class method.

        """
        if not self.is_limits_base:
            setattr(self.args, 'dont_write_raytrace_limits', True)
        super(RayTraceTask, self).reset_outputs(**kwargs)

    @classmethod
    def _output_suffix(cls, args, name, wildcards=None):
        r"""Generate the suffix containing information about parameters
        that should be added to generated output files.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Base name for variable to set.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.

        Returns:
            str: Suffix.

        """
        suffix = ''
        if name == 'traced_mesh':
            suffix += cls._make_suffix(
                args, 'query', wildcards=wildcards, cond=True,
            )
            suffix += cls._make_suffix(
                args, 'show_rays', value='rays', wildcards=wildcards,
            )
            suffix += cls._make_suffix(
                args, 'highlight', prefix='_highlight', title=True,
                wildcards=wildcards,
            )
            return suffix
        elif name != 'raytrace':
            return suffix
        suffix += cls._make_suffix(
            args, 'time', wildcards=wildcards,
        )
        suffix += cls._make_suffix(
            args, 'nrays', cond=True, wildcards=wildcards,
        )
        suffix += cls._make_suffix(
            args, 'multibounce', value='multibounce',
            wildcards=wildcards,
        )
        suffix += cls._make_suffix(
            args, 'any_direction', value='anydirection',
            wildcards=wildcards,
        )
        suffix += cls._make_suffix(
            args, 'periodic_canopy_count', prefix='_periodic',
            cond=bool(args.periodic_canopy), wildcards=wildcards,
        )
        suffix += cls._make_suffix(
            args, 'periodic_canopy', wildcards=wildcards,
        )
        return suffix

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
            face_values = query_values[query]
        else:
            face_values = (
                query_values['flux_density'] * query_values['areas']
            )
        return face_values

    @classmethod
    def extract_query_totals(cls, values, per_plant=False):
        r"""Calculate sums over all query options.

        Args:
            values (dict): Query values for each face.
            per_plant (bool, optional): If True, the query values should
                also be totaled for each plant. If an int is provided,
                the number of unique plant IDs should match the provided
                value.

        """
        values['flux'] = cls.extract_query(values, 'flux')

        def sum(x):
            if isinstance(x, units.QuantityArray):
                return units.Quantity(x.value.sum(), x.units)
            return x.sum()

        out = {k: {'total': sum(values[k])} for k in _query_options}
        if per_plant:
            plantids = values['plantids']
            plantids_unique = np.unique(plantids)
            if isinstance(per_plant, int):
                assert len(plantids_unique) == per_plant
            for i in plantids_unique:
                idx = (plantids == i)
                for k in _query_options:
                    out[k][i] = sum(values[k][idx])
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
    def _get_base_id(cls, args):
        generator_class = get_class_registry().get('crop', args.crop)
        return generator_class.ids_from_file(args.data)[0]

    @classmethod
    def _adjust_color_limits(cls, self, name=None):
        r"""Set the minimum and maximum for color mapping.

        Args:
            self (TaskBase): Task that is running.
            name (str, optional): Name for limits to set if different
                than raytrace.

        """
        if name is None:
            name = self._name
        var_min = f'color_vmin_{name}'
        var_max = f'color_vmax_{name}'
        if ((getattr(self.args, var_min) is not None
             and getattr(self.args, var_max) is not None)):
            return
        limits = self.get_output('raytrace_limits')
        cls.update_color_limits(self.args, limits=limits, name=name)

    @classmethod
    def update_color_limits(cls, args, limits=None, name=None,
                            force=False, **kwargs):
        r"""Set the minimum and maximum for color mapping.

        Args:
            args (argparse.Namespace): Parsed arguments.
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
        var_min = f'color_vmin_{name}'
        var_max = f'color_vmax_{name}'
        var_scaling = f'colormap_scaling_{name}'
        if (((not force)
             and getattr(args, var_min) is not None
             and getattr(args, var_max) is not None)):
            return False
        out = False
        if limits is None:
            limits = RayTraceTask._generate_limits_class(args, **kwargs)
            out = True
        vscaling = getattr(args, var_scaling)
        limits = limits[args.query]
        if force or getattr(args, var_min) is None:
            setattr(args, var_min, limits[f'vmin_{vscaling}'])
        if force or getattr(args, var_max) is None:
            setattr(args, var_max, limits[f'vmax_{vscaling}'])
        cls.log_class(f'LIMITS[{cls._name}, {name}]: '
                      f'{getattr(args, var_min)}, '
                      f'{getattr(args, var_max)}')
        return out

    def _color_scene(self):
        r"""Color the geometry based on the raytracer results."""
        query_values = self.get_output('raytrace')
        if self.args.show_rays:
            inst = self.run_iteration(
                output_name='instance',
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
            self.set_output('traced_mesh', mesh)
            return
        self._adjust_color_limits(self)
        mesh = self.mesh
        face_values = self.extract_query(query_values, self.args.query)
        if isinstance(face_values, units.QuantityArray):
            face_values = np.array(face_values)
        vertex_values = self.raytracer.face2vertex(
            face_values, method='deposit')
        vertex_colors = utils.apply_color_map(
            vertex_values,
            color_map=self.args.colormap_raytrace,
            vmin=self.args.color_vmin_raytrace,
            vmax=self.args.color_vmax_raytrace,
            scaling=self.args.colormap_scaling_raytrace,
            highlight=self.args.highlight,
            highlight_color=self.args.highlight_color.value,
        )
        mesh.add_colors('vertex', vertex_colors)
        self.set_output('traced_mesh', mesh)

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
            import pdb
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
        try:
            self.raytracer.args.query = query
            out = self.raytracer.raytrace()
        finally:
            self.raytracer.args.query = query0
        return out

    def _raytrace_scene(self):
        r"""Run the ray tracer on the selected geometry.

        Returns:
            dict: Dictionary of ray tracer queries.

        """
        values = {}
        for k in _query_options:
            if k == 'flux':  # calculated from flux_density & areas
                continue
            query0 = self.args.query
            # self.cache_args(args_overwrite={'query': k},
            #                 properties_preserve=['raytracer'],
            #                 recursive=False)
            try:
                self.raytracer.args.query = k
                # self.raytracer.args = self.args
                values[k] = self.raytracer.raytrace()
                # self.restore_args()
                # self.raytracer.args = self.args
            finally:
                self.raytracer.args.query = query0
        self.set_output('raytrace', values)

    @classmethod
    def _generate_limits_class(cls, args, base_limits_args=None,
                               **kwargs):
        r"""Generate limits for the date in question.

        Args:
            args (argparse.Namespace): Parsed arguments.
            base_limits_args (dict): Arguments to use to generate base
                limits.
            **kwargs: Additional keyword arguments are passed to
                run_iteration_class.

        Returns:
            dict: Limits.

        """
        if base_limits_args is None:
            base_limits_args = cls.base_limits_args_class(args)
        args_overwrite = dict(
            base_limits_args,
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
        if not self.is_limits_base:
            return self._generate_limits_class(
                self.args, base_limits_args=self.base_limits_args,
                root=self.root, cache_outputs=['raytrace_limits'],
            )
        return self.calc_query_limits(self.get_output('raytrace'))

    @property
    def is_limits_base(self):
        r"""bool: True if the current arguments are what is required for
        calculating limits."""
        return (self.limits_args == self.base_limits_args)

    @classmethod
    def base_limits_args_class(cls, args, base_id=None):
        r"""dict: Arguments that should be used to generate base limits.
        """
        out = {
            'time': 'noon',
            'date': args.time.summer_solstice,
        }
        if not args.output_generate.generated:
            return out
        if base_id is None:
            base_id = RayTraceTask._get_base_id(args)
        out.update(
            id=base_id,
            age='maturity',
        )
        return out

    @property
    def base_limits_args(self):
        r"""dict: Arguments that should be used to generate base limits.
        """
        base_id = (
            None if (not self.args.output_generate.generated)
            else self.external_tasks['generate'].all_ids[0]
        )
        return self.base_limits_args_class(self.args, base_id=base_id)

    @property
    def limits_args(self):
        r"""dict: Current arguments that should be compared against
        base_limits_args."""
        out = {
            'time': self.args.time.solar_time_string,
            'date': self.args.time.date,
        }
        if not self.args.output_generate.generated:
            return out
        out.update(
            id=self.args.id,
            age=self.args.time.crop_age_string,
        )
        return out

    def generate_output(self, name):
        r"""Generate the specified output value.

        Args:
            name (str): Name of the output to generate.

        Returns:
            object: Generated output.

        """
        if ((self.args.id == 'all_split'
             and self.args.output_generate.generated
             and not self.output_exists('generate'))):
            over = {'id': self.external_tasks['generate'].all_ids}
            mesh = None
            x = 0.0 * self.args.x
            y = 0.0 * self.args.y
            plantid = 0
            for inst in self.run_series(self.args, over=over,
                                        output_name='instance'):
                if self.output_enabled('traced_mesh'):
                    if mesh is None:
                        mesh = inst.get_output('traced_mesh')
                    else:
                        mesh.append(self.shift_mesh(
                            inst.get_output('traced_mesh'),
                            x, y, plantid=plantid,
                        ))
                    x += self.args.row_spacing * (self.args.nrows + 2)
                    plantid += (self.args.nrows * self.args.ncols)
            if self.output_enabled('traced_mesh'):
                assert mesh is not None
                self.set_output('traced_mesh', mesh)
            self.set_output('raytrace', None)
            self.set_output('raytrace_limits', None)
            return
        if name == 'raytrace':
            self._raytrace_scene()
        elif name == 'raytrace_limits':
            out = self._generate_limits()
            self.set_output('raytrace_limits', out)
        elif name == 'traced_mesh':
            self._color_scene()
        else:
            super(RayTraceTask, self).generate_output(name)


class RenderTask(TaskBase):
    r"""Class for rendering a 3D canopy."""

    _name = 'render'
    _output_info = {
        'render': {
            'ext': '.png',
            'base': 'raytrace',
            'description': 'the rendered image',
        },
        'render_camera': {
            'ext': '.json',
            'base': 'generate',
            'description': 'camera properties',
            'optional': True,
        },
    }
    _external_tasks = {
        RayTraceTask: {
            'exclude': [
                'show_rays', 'ray_color', 'ray_width', 'ray_length',
                'arrow_width', 'highlight', 'highlight_color',
            ],
            'modifications': {
                'colormap': {'default': 'viridis'},
                'colormap_scaling': {'default': 'log'},
            }
        },
    }
    _composite_arguments = {
        'age': {
            'scene_age': {
                'description': (
                    'that the camera position should be calculated for '
                    '(only valid for generated meshes)'
                ),
                'ignore': ['planting_date'],
                'optional': True,
            },
        },
        'color': {
            'background_color': {
                'description': 'that should be used for the scene',
                'defaults': {'color': 'transparent'},
            }
        },
    }
    _argument_conversions = {
        'mesh_units': [
            'image_width', 'image_height',
        ],
    }
    _arguments = [
        (('--camera-type', ), {
            'type': str,
            'choices': ['projection', 'orthographic'],  # 'spherical'],
            'default': 'projection',
            'help': ('Type of camera that should be used to render the '
                     'scene'),
        }),
        (('--scene-mins', ), {
            # 'type': parse_axis,
            'help': 'Minimum extent of scene along each dimension',
        }),
        (('--scene-maxs', ), {
            # 'type': parse_axis,
            'help': 'Maximum extent of scene along each dimension',
        }),
        (('--camera-direction', ), {
            'type': str,
            'help': ('Direction that camera should face. If not '
                     'provided, the camera will point to the center of '
                     'the scene from its location.'),
        }),
        (('--camera-fov-width', ), {
            'type': parse_quantity, 'units': 'degrees', 'default': 45.0,
            'help': ('Angular width of the camera\'s field of view (in '
                     'degrees) for a projection camera.'),
        }),
        (('--camera-fov-height', ), {
            'type': parse_quantity, 'units': 'degrees', 'default': 45.0,
            'help': ('Angular height of the camera\'s field of view (in '
                     'degrees) for a projection camera.'),
        }),
        (('--camera-up', ), {
            'type': str,
            'help': ('Up direction for the camera. If not provided, the '
                     'up direction for the scene will be assumed.'),
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
        }),
        (('--image-nx', ), {
            'type': int,
            'help': ('Number of pixels for the rendered image in the '
                     'horizontal direction. If not provided, but '
                     '--image-ny is provided, the value for --image-nx '
                     'will be determined from --image-ny by assuming '
                     'a constant resolution in both directions. If '
                     'neither are provided, --image-ny defaults to '
                     '1024.'),
        }),
        (('--image-ny', ), {
            'type': int,
            'help': ('Number of pixels for the rendered image in the '
                     'vertical direction. If not provided, but '
                     '--image-nx is provided, the value for --image-ny '
                     'will be determined from --image-nx by assuming '
                     'a constant resolution in both directions. If '
                     'neither are provided, --image-ny defaults to '
                     '1024.'),
        }),
        (('--image-width', ), {
            'type': parse_quantity, 'units': 'cm',
            'help': ('Width of the image (in cm). If not provided, '
                     'the width will be set based on the camera '
                     'position and type such that the entire scene '
                     'is captured.'),
        }),
        (('--image-height', ), {
            'type': parse_quantity, 'units': 'cm',
            'help': ('Height of the image (in cm). If not provided, '
                     'the height will be set based on the camera '
                     'position and type such that the entire scene '
                     'is captured.'),
        }),
        (('--resolution', ), {
            'type': parse_quantity, 'units': 'cm**-1',  # 'default': 5,
            'help': ('Resolution that the scene should be rendered with '
                     'in pixels per centimeter. If provided, any '
                     'values provided for --image-nx and --image-ny '
                     'will be ignored. If not provided, the resolution '
                     'in each direction will be determined by '
                     '--image-nx and --image-ny.'),
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
        if name == 'render_camera':
            with open(fname, 'r') as fd:
                return rapidjson.load(fd)
        return utils.read_png(fname, verbose=args.verbose)

    @classmethod
    def _write_output(cls, args, name, fname, output):
        r"""Write to an output file.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Name of the output to write.
            fname (str): Path of the file that should be written to.
            output (object): Output object to write to file.

        """
        if name == 'render_camera':
            with open(fname, 'w') as fd:
                rapidjson.dump(output, fd,
                               write_mode=rapidjson.WM_PRETTY)
            return
        utils.write_png(output, fname, verbose=args.verbose)

    @classmethod
    def adjust_args_internal(cls, args, **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        args.show_rays = False
        if ((args.camera_direction is None
             and args.camera_location is None)):
            args.camera_direction = 'downnortheast'
        super(RenderTask, cls).adjust_args_internal(args, **kwargs)

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

    def reset_outputs(self, **kwargs):
        r"""Remove existing files that should be overwritten.

        Args:
            **kwargs: Keyword arguments are passed to base class method.

        """
        if not self.is_camera_base:
            setattr(self.args, 'dont_write_render_camera', True)
        super(RenderTask, self).reset_outputs(**kwargs)

    @classmethod
    def _output_suffix(cls, args, name, wildcards=None):
        r"""Generate the suffix containing information about parameters
        that should be added to generated output files.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Base name for variable to set.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.

        Returns:
            str: Suffix.

        """
        suffix = ''
        if name != 'render_camera':
            suffix += cls._make_suffix(
                args, 'query', wildcards=wildcards, cond=True,
            )
        suffix += cls._make_suffix(
            args, 'camera_direction', wildcards=wildcards,
        )
        suffix += cls._make_suffix(
            args, 'camera_location', prefix='_from_',
            wildcards=wildcards,
        )
        suffix += cls._make_suffix(
            args, 'scene_age', wildcards=wildcards,
        )
        suffix += cls._make_suffix(
            args, 'camera_type', noteq='projection',
            wildcards=wildcards,
        )
        if args.camera_type == 'projection':
            suffix += cls._make_suffix(
                args, 'camera_fov_width', conv=int, prefix='',
                wildcards=wildcards,
            )
            suffix += cls._make_suffix(
                args, 'camera_fov_height', conv=int, prefix='x',
                wildcards=wildcards,
            )
        suffix += cls._make_suffix(
            args, 'camera_up', prefix='up', wildcards=wildcards,
        )
        suffix += cls._make_suffix(
            args, 'background_color', noteq='transparent',
            wildcards=wildcards,
        )
        suffix += cls._make_suffix(
            args, 'image_nx', wildcards=wildcards,
        )
        suffix += cls._make_suffix(
            args, 'image_ny', prefix='x', wildcards=wildcards,
        )
        suffix += cls._make_suffix(
            args, 'image_width', conv=int,
            wildcards=wildcards,
        )
        suffix += cls._make_suffix(
            args, 'image_height', conv=int, prefix='x', suffix='cm',
            wildcards=wildcards,
        )
        suffix += cls._make_suffix(
            args, 'resolution', conv=int, suffix='percm',
            wildcards=wildcards,
        )
        suffix += cls._make_suffix(
            args, 'scene_mins', wildcards=wildcards,
        )
        suffix += cls._make_suffix(
            args, 'scene_maxs', wildcards=wildcards,
        )
        return suffix

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
        RayTraceTask._adjust_color_limits(self)
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
            color_map=self.args.colormap_render,
            vmin=self.args.color_vmin_render,
            vmax=self.args.color_vmax_render,
            scaling=self.args.colormap_scaling_render,
            highlight=(pixel_values < 0),
            highlight_color=self.args.background_color.value,
            include_alpha=(len(self.args.background_color.value) == 4)
        )
        return image

    @property
    def is_camera_base(self):
        r"""bool: True if the current arguments are what is required for
        calculating camera properties."""
        return (self.camera_args == self.base_camera_args)

    @classmethod
    def base_camera_args_class(cls, args):
        r"""dict: Arguments that should be used to generate base camera
        properties."""
        out = {}
        if not args.output_generate.generated:
            return out
        out.update(
            age='maturity',
            scene_age=None,
        )
        if args.scene_age.args['age']:
            out['age'] = args.scene_age.args['age']
        return out

    @property
    def base_camera_args(self):
        r"""dict: Arguments that should be used to generate base camera
        properties."""
        return self.base_camera_args_class(self.args)

    @property
    def camera_args(self):
        r"""dict: Current arguments that should be compared against
        base_camera_args."""
        out = {}
        if not self.args.output_generate.generated:
            return out
        out.update(
            age=self.args.time.args['age'],
            scene_age=self.args.scene_age.args['age'],
        )
        return out

    @classmethod
    def _generate_camera_class(cls, args, base_camera_args=None,
                               **kwargs):
        r"""Generate camera properties for a specific scene age.

        Args:
            args (argparse.Namespace): Parsed arguments.
            base_camera_args (dict): Arguments to use to generate base
                camera properties.
            **kwargs: Additional keyword arguments are passed to
                run_iteration_class.

        Returns:
            dict: Camera properties.

        """
        if base_camera_args is None:
            base_camera_args = cls.base_camera_args_class(args)
        args_overwrite = dict(
            base_camera_args,
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
        if not self.is_camera_base:
            return self._generate_camera_class(
                self.args, base_camera_args=self.base_camera_args,
                root=self.root, cache_outputs=['render_camera'],
            )
        out = {}
        for k in ['camera_type', 'camera_fov_width',
                  'camera_fov_height']:
            out[k] = getattr(self.args, k)
        for k in ['camera_direction', 'camera_up', 'camera_location',
                  'image_nx', 'image_ny',
                  'image_width', 'image_height',
                  'resolution', 'scene_mins', 'scene_maxs']:
            out[k] = getattr(self.raytracer, k)
        return out

    # def _adjust_camera_scene_age(self):
    #     r"""Set camera attributes for a specific scene age."""
    #     assert self.args.scene_age.age is not None
    #     assert self.args.output_render.path
    #     age_args = {
    #         'age': self.args.scene_age.args['age'],
    #         'output_render': False,
    #         'output_render_camera': True,
    #         'scene_age': None,
    #         'scene_mins': None,
    #         'scene_maxs': None,
    #         'time': None,
    #     }
    #     print(80 * '-')
    #     print(f"GENERATING CAMERA PROPERTIES FOR "
    #           f"{self.args.scene_age.age}")
    #     pprint.pprint(age_args)
    #     camera_args = self.run_iteration(
    #         args_overwrite=age_args,
    #         output_name='render_camera',
    #         cache_outputs=['render_camera'],
    #     )
    #     pprint.pprint(camera_args)
    #     for k, v in camera_args.items():
    #         setattr(self.args, k, v)
    #     print("END GENERATING CAMERA PROPERTIES")
    #     print(80 * '-')
    #     return camera_args

    def generate_output(self, name):
        r"""Generate the specified output value.

        Args:
            name (str): Name of the output to generate.

        Returns:
            object: Generated output.

        """
        if name not in ['render', 'render_camera']:
            return super(RenderTask, self).generate_output(name)
        if ((self.args.id == 'all_split'
             and self.args.output_generate.generated
             and not self.output_exists(self.output_file('generate')))):
            over = {'id': self.external_tasks['generate'].all_ids}
            images = []
            for x in self.run_series(self.args, over=over,
                                     output_name=name):
                images.append(x)
            if name == 'render':
                image = np.concatenate(images, axis=1)
                self.set_output('render', image)
            else:
                self.set_output('render_camera', None)
            return
        if name == 'render_camera':
            out = self._generate_camera()
            self.set_output(name, out)
            return
        if self.args.scene_age.age is not None:
            before = self.args.output_render.path
            camera_args = self.get_output('render_camera')
            for k, v in camera_args.items():
                setattr(self.args, k, v)
            after = self.args.output_render.path
            assert after == before
        image = self._render_scene()
        self.set_output('render', image)


class TotalsTask(TemporalTaskBase):
    r"""Class for plotting the flux on a geometry as a function of time."""
    _name = 'totals'
    _step_task = RayTraceTask
    _output_info = {
        'totals': {
            'ext': '.json',
            'description': 'raytraced query totals',
        },
        'totals_plot': {
            'ext': '.png',
            'base': 'totals',
            'description': 'a plot of raytraced query totals',
            'optional': True,
        },
    }
    _arguments = [
        (('--per-plant', ), {
            'action': 'store_true',
            'help': ('Calculate the totals on a per-plant basis'),
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
        if name == 'totals':
            with open(fname, 'r') as fd:
                out = rapidjson.load(fd)
            out['times'] = [
                datetime.fromisoformat(x) for x in out['times']
            ]
            return out
        # elif name == 'totals_plot':
        #     return utils.read_png(fname, verbose=args.verbose)
        return super(TotalsTask, cls)._read_output(args, name, fname)

    @classmethod
    def _write_output(cls, args, name, fname, output):
        r"""Write to an output file.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Name of the output to write.
            fname (str): Path of the file that should be written to.
            output (object): Output object to write to file.

        """
        if name == 'totals_plot':
            output.savefig(fname, dpi=300)
            return
        elif name == 'totals':
            output = dict(output,
                          times=[x.isoformat() for x in output['times']])
            with open(fname, 'w') as fd:
                rapidjson.dump(output, fd,
                               write_mode=rapidjson.WM_PRETTY)
            return
        super(TotalsTask, cls)._write_output(args, name, fname, output)

    @classmethod
    def _output_suffix(cls, args, name, wildcards=None):
        r"""Generate the suffix containing information about parameters
        that should be added to generated output files.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Base name for variable to set.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.

        Returns:
            str: Suffix.

        """
        suffix = ''
        if name == 'totals_plot':
            return suffix
        suffix += super(TotalsTask, cls)._output_suffix(
            args, name, wildcards=wildcards,
        )
        suffix += cls._make_suffix(
            args, 'per_plant', value='per_plant',
            wildcards=wildcards
        )
        return suffix

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

    def plot_data(self, times, totals, id=None, iclass=0):
        r"""Plot the data for a single crop ID.

        Args:
            times (list): Times for the data in totals.
            totals (dict): Mapping between query name and dictionaries
                of totals for each component.
            id (str, optional): ID of the crop that should be used for
                the label.
            iclass (int, optional): Index of the ID that should be used
                to select the color.

        """
        if id is None:
            id = self.args.id
            iclass = 0
        totals = totals[self.args.query]
        first = getattr(self, 'first', True)
        ax = self.axes
        linestyles = ['-', ':']
        colors = ['blue', 'orange']
        if first:
            ylabel = self.args.query.title()  # TODO: Add units
            if isinstance(totals['total'], units.QuantityArray):
                ylabel += f" ({totals['total'].units})"
            elif isinstance(totals['total'][0], units.Quantity):
                ylabel += f" ({totals['total'][0].units})"
            ax.set_xlabel('Time')
            ax.set_ylabel(ylabel)
        if self.args.per_plant:
            lines = {'interior': {}, 'exterior': {}}
            # TODO: Split plants by crop class when they are
            # combined in the same mesh
            for k, v in totals.items():
                if k == 'total':
                    continue
                loc = None
                locStr = None
                label = None
                if self.isExteriorPlant(k):
                    loc = 1
                    locStr = 'exterior'
                else:
                    loc = 0
                    locStr = 'interior'
                lines[locStr][k] = v
                if len(lines[locStr]) == 1:
                    label = f'{locStr.title()} plants ({id})'
                color = colors[iclass]
                style = linestyles[loc]
                ax.plot(times, v, label=label, color=color,
                        linestyle=style)
            # print(f'INTERIOR PLANTS [{id}]: '
            #       f'{len(lines["interior"])}/{len(totals) - 1}')
            # print(f'EXTERIOR PLANTS [{id}]: '
            #       f'{len(lines["exterior"])}/{len(totals) - 1}')
        else:
            color = colors[iclass]
            style = linestyles[0]
            ax.plot(times, totals['total'], color=color,
                    label=f'total ({id})',
                    linestyle=style)

    def finalize_step(self, x):
        r"""Finalize the output from a step.

        Args:
            x (object): Result of step.

        Returns:
            object: Finalized step result.

        """
        values = x.get_output('raytrace')
        if not RayTraceTask._check_for_plantids(self, values):
            assert not x.args.overwrite_raytrace
            print("REGENERATING RAY TRACE WITHOUT PLANTIDS")
            raise RepeatIterationError(args_overwrite={
                'overwrite_raytrace': True})
        per_plant = ((self.args.nrows * self.args.ncols)
                     if self.args.per_plant else False)
        return (
            x.args.id, x.args.time.time,
            RayTraceTask.extract_query_totals(values, per_plant=per_plant),
        )

    def join_steps(self, xlist):
        r"""Join the output form all of the steps.

        Args:
            xlist (list): Result of all steps.

        Returns:
            object: Joined output from all steps.

        """
        if self.args.id == 'all_split':
            idlist = self.external_tasks['generate'].all_ids
        else:
            idlist = sorted(list(set([x[0] for x in xlist])))
        # print(xlist[0])
        # import pdb; pdb.set_trace()
        times = sorted(list(set([x[1] for x in xlist])))
        out = {id: {k: {i: [] for i in ids.keys()}
                    for k, ids in xlist[0][-1].items()}
               for id in idlist}
        for id, time, x in xlist:
            for k, ids in x.items():
                for i, v in ids.items():
                    out[id][k][i].append(v)
        out['times'] = times
        return out

    def generate_output(self, name):
        r"""Generate the specified output value.

        Args:
            name (str): Name of the output to generate.

        Returns:
            object: Generated output.

        """
        if name == 'totals_plot':
            out = self.get_output('totals')
            if self.args.id == 'all_split':
                idlist = self.external_tasks['generate'].all_ids
            else:
                idlist = sorted([k for k in out.keys() if k != 'times'])
            for iclass, id in enumerate(idlist):
                self.plot_data(out['times'], out[id], id=id,
                               iclass=iclass)
            self.axes.legend()
            self.set_output('totals_plot', self.figure)
            return
        super(TotalsTask, self).generate_output(name)

    def step_args(self):
        r"""Yield the updates that should be made to the arguments for
        each step.

        Yields:
            dict: Step arguments.

        """
        if self.args.id == 'all_split':
            for id in self.external_tasks['generate'].all_ids:
                for args_overwrite in super(TotalsTask, self).step_args():
                    yield dict(args_overwrite, id=id)
            return
        for args_overwrite in super(TotalsTask, self).step_args():
            yield args_overwrite


class AnimateTask(TemporalTaskBase):
    r"""Class for producing an animation."""

    _name = 'animate'
    _step_task = RenderTask
    _output_info = {
        'animate': {
            'base': 'render',
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
        },
    }
    _arguments = [
        (('--movie-format', ), {
            'type': str, 'choices': ['mp4', 'mpeg', 'gif'],
            'default': 'gif',
            'help': 'Format that the movie should be output in',
        }),
        (('--frame-rate', ), {
            'type': int, 'default': 1,
            'help': ('The frame rate that should be used for the '
                     'generated movie in frames per second'),
        }),
        (('--inset-totals', ), {
            'action': 'store_true',
            'help': ('Inset a plot of the query total below the '
                     'image.'),

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
    def adjust_args_internal(cls, args, **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        for k in ['colormap', 'color_vmin', 'color_vmax',
                  'colormap_scaling']:
            setattr(args, f'{k}_render', getattr(args, f'{k}_animate'))
        super(AnimateTask, cls).adjust_args_internal(args, **kwargs)

    @classmethod
    def _output_suffix(cls, args, name, wildcards=None):
        r"""Generate the suffix containing information about parameters
        that should be added to generated output files.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Base name for variable to set.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.

        Returns:
            str: Suffix.

        """
        assert name == cls._name
        suffix = ''
        suffix += super(AnimateTask, cls)._output_suffix(
            args, name, wildcards=wildcards,
        )
        suffix += cls._make_suffix(
            args, 'inset_totals', value='totals', wildcards=wildcards,
        )
        # suffix += cls._make_suffix(
        #     args, 'per_plant_totals', value='per_plant',
        #     wildcards=wildcards
        # )
        suffix += cls._make_suffix(
            args, 'frame_rate', suffix='fps', noteq=1,
            wildcards=wildcards,
        )
        return suffix

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
        return cls._make_suffix(
            args, 'movie_format', prefix='.', cond=True,
            wildcards=wildcards,
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
    _arguments = [
        (('--vary', ), {
            'type': str,
            'help': 'Argument that should be varied.',
            'default': 'row_spacing',
        }),
        (('--goal-id', ), {
            'help': 'ID that the Goal of the optimization.',
        }),
        (('--goal', ), {
            'help': 'Goal of the optimization.',
            'default': 'flux',
            'choices': _query_options,
        }),
    ]

    @classmethod
    def adjust_args_internal(cls, args):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        if args.goal_id is None:
            args.goal_id = RayTraceTask._get_base_id(args)
            # args.goal_id = self.external_tasks['generate'].all_ids[0]
        super(MatchQuery, cls).adjust_args_internal(args)
        assert args.id != args.goal_id

    @cached_property
    def goal(self):
        r"""units.Quantity: Value that should be achieved."""
        args_overwrite = {
            'id': self.args.goal_id,
            self.args.vary: None,
        }
        inst = self.run_iteration(
            output_name='instance',
            args_overwrite=args_overwrite,
        )
        return self.finalize_step(inst)

    def finalize_step(self, x):
        r"""Finalize the output from a step.

        Args:
            x (object): Result of step.

        Returns:
            object: Finalized step result.

        """
        # TODO: Only run requested query
        if x.output_exists('raytrace'):
            return x.get_output('raytrace')[self.args.goal]
