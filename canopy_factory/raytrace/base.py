import numpy as np
import itertools
from collections import OrderedDict
from yggdrasil_rapidjson import units
from yggdrasil_rapidjson.geometry import Ply as PlyDict
from yggdrasil_rapidjson.geometry import ObjWavefront as ObjDict
from canopy_factory import utils
from canopy_factory.utils import (
    RegisteredClassBase,
    parse_quantity, parse_axis,
    cached_property, cached_args_property,
)


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
        xx = np.meshgrid(*[limits[:, i] for i in range(limits.shape[1])])
        out = np.vstack([ixx.flatten() for ixx in xx]).T
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

    @property
    def plant_triangles(self):
        r"""Yields tuples of plant ID & arrays of triangles for the
        faces in that plant."""
        for plantid, mesh_dict in self.plants.items():
            triangles = []
            for face in mesh_dict['face']:
                triangles.append(mesh_dict['vertex'][face, :])
            if triangles:
                yield plantid, np.array(triangles)

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
    def finalize_face_index(cls, idx, Nface, shift=0):
        r"""Convert a face index to a numpy array.

        Args:
            idx (dict): Dictionary of indices created when selecting a
                subset of a mesh.
            Nface (int): Number of faces in the mesh.
            shift (int, optional): Factor that should be added to the
                final index.

        Returns:
            np.ndarray: Face indices selected by idx.

        """
        idx_chain = []
        iidx = idx
        while iidx is not None:
            idx_chain.append(iidx['face'])
            iidx = iidx.get('parent', None)
        idx_dst = np.arange(Nface, dtype=np.int32)
        for iidx in idx_chain[::-1]:
            assert idx_dst.shape == iidx.shape
            idx_dst = idx_dst[iidx]
        return idx_dst + shift

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
    def nvirtual(self):
        if not self.args.canopy.startswith('virtual'):
            return 0
        return np.prod(self.args.virtual_canopy_count_array) - 1

    @cached_args_property
    def virtual_shifts(self):
        r"""np.ndarray: Shifts for positions of virtual plants."""
        if not self.args.canopy.startswith('virtual'):
            return np.zeros((0, 3), 'f4')
        return utils.get_periodic_shifts(
            self.args.virtual_period,
            self.args.virtual_direction,
            self.args.virtual_canopy_count_array,
            dont_reflect=True,
            dont_center=(self.args.canopy != 'virtual_single'),
        ).astype('f4')

    @cached_args_property
    def periodic_shifts(self):
        r"""np.ndarray: Shifts for positions of periodic plants."""
        if self.args.periodic_canopy not in ['scene', 'plants']:
            return np.zeros((0, 3), 'f4')
        return utils.get_periodic_shifts(
            self.args.periodic_period,
            self.args.periodic_direction,
            self.args.periodic_canopy_count_array,
        ).astype('f4')

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
        if not self.args.canopy.startswith('virtual'):
            return self.real_scene_model
        mins = self.real_scene_model.mins
        maxs = self.real_scene_model.maxs
        shifts = np.vstack([np.zeros(mins.shape, 'f4'),
                            self.virtual_shifts])
        if getattr(self.args, 'scene_mins', None) is None:
            mins = (mins + shifts).min(axis=0)
        if getattr(self.args, 'scene_maxs', None) is None:
            maxs = (maxs + shifts).max(axis=0)
        return SceneModel(mins, maxs)

    @cached_args_property
    def field_scene_model(self):
        r"""SceneModel: Field scene properties."""
        mins = self.geometryids['HEADER_JSON']['field_mins']
        maxs = self.geometryids['HEADER_JSON']['field_maxs']
        if self.args.canopy.startswith('virtual'):
            shifts = utils.project_onto_ground(
                np.vstack([
                    np.zeros((self.virtual_shifts.shape[1]), 'f4'),
                    self.virtual_shifts]),
                self.args.axis_rows, self.args.axis_cols,
            )
            scene_units = mins.units
            mins = parse_quantity(
                (mins.value + shifts).min(axis=0), scene_units)
            maxs = parse_quantity(
                (maxs.value + shifts).max(axis=0), scene_units)
        return SceneModel(mins, maxs)

    @cached_args_property
    def layout_scene_model(self):
        r"""SceneModel: Field scene properties."""
        mins = (
            self.args.x * self.args.axis_rows
            + self.args.y * self.args.axis_cols
        )
        maxs = (
            mins
            + self.args.plot_width * self.args.axis_rows
            + self.args.plot_length * self.args.axis_cols
        )
        if self.args.canopy.startswith('virtual'):
            shifts = np.vstack([np.zeros(mins.shape, 'f4'),
                                self.virtual_shifts])
            mins = (mins + shifts).min(axis=0)
            maxs = (maxs + shifts).max(axis=0)
        return SceneModel(mins, maxs)

    @cached_args_property
    def ground(self):
        r"""np.ndarray: Coordinates of the scene center on the ground."""
        return (
            np.dot(self.virtual_scene_model.dim / 2, self.north) * self.north
            + np.dot(self.virtual_scene_model.dim / 2, self.east) * self.east
            + self.args.ground_height.value * self.up
        )

    @cached_args_property
    def ground_origin(self):
        r"""np.ndarray: Coordinates of the scene center on the ground."""
        return (
            np.dot(self.virtual_scene_model.center, self.north) * self.north
            + np.dot(self.virtual_scene_model.center, self.east) * self.east
            + self.args.ground_height.value * self.up
        )

    @cached_args_property
    def zenith(self):
        r"""np.ndarray: Vector from the ground in the up direction with
        a length equal to the farthest point in the scene from the
        ground"""
        return (
            self.up * np.sqrt(
                np.max(np.sum((self.virtual_scene_model.limits
                               - self.ground)**2, axis=1)))
            + self.ground
        )

    @cached_args_property
    def zenith_origin(self):
        r"""np.ndarray: Vector from the ground in the up direction with
        a length equal to the farthest point in the scene from the
        ground"""
        return (
            self.up * np.sqrt(
                np.max(np.sum((self.virtual_scene_model.limits
                               - self.ground_origin)**2, axis=1)))
            + self.ground_origin
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

    def raytrace(self, query):
        r"""Run the ray tracer and get values for each face.

        Args:
            query (str): Type of raytrace operation that should be
                performed.

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

    def coverage(self):
        r"""Compute the scene coverage.

        Returns:
            float: Percent of rays that hit the plants.

        """
        raise NotImplementedError

    def face2vertex(self, face_scalar, method='average'):
        r"""Convert an array of scalars for each face to an array of
        scalars for each vertex.

        Args:
            face_scalar (np.ndarray): Array of scalars for each face.
            method (str, optional): Method to use to map from face values
                to vertex values.::

                    'average': Average over the values for each face that
                        vertices are part of.
                    'deposit': Split the values for each face amongst its
                        vertices additively.

        Returns:
            np.ndarray: Array of scalars for each vertex.

        """
        faces = self.mesh_dict['face'][self.area_mask, :]
        nvert = self.mesh_dict['vertex'].shape[0]
        area_mask = self.area_mask
        if self.args.canopy == 'virtual':
            face_scalar = face_scalar[:len(self.area_mask)]
        face_scalar = face_scalar[area_mask]
        if method == 'deposit':
            face_scalar /= faces.shape[1]
        vertex_scalar = np.zeros((nvert, ))
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
