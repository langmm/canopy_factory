import os
import pdb
import numpy as np
from collections import OrderedDict
from datetime import datetime
from yggdrasil import units
from yggdrasil.serialize.PlySerialize import PlyDict
from yggdrasil.serialize.ObjSerialize import ObjDict
from canopy_factory import utils
from canopy_factory.utils import (
    RegisteredClassBase, get_class_registry,
    parse_quantity, parse_axis, parse_color,
    cached_property, cached_args_property, readonly_cached_args_property,
    # Geometry
    scene2geom, _lpy_rays,
)
from canopy_factory.cli import TemporalTaskBase
from canopy_factory.crops import LayoutTask, GenerateTask


_query_options = ['flux_density', 'flux', 'hits', 'areas', 'plantids']


class SolarModel(object):
    r"""Solar model using pvlib. For quantities with units, values can
    be provided as floats (in which case the default units will be
    assumed) or units.Quantity instances.

    Args:
        latitude (float): Location latitude (in degrees).
        longitude (float): Location latitude (in degrees).
        time (datetime.datetime): Time.
        altitude (float, optional): Altitude (in meters) used to compute
            solar position. If not provided, but pressure is, pressure
            will be used to calculate altitude.
        pressure (float, optional): Pressure (in Pa) used to compute
            solar position. If not provided, but altitude is, altitude
            will be used to calculate pressure.
        temperature (float, optional): Air temperature (in degrees C)
            used to compute solar position.
        eta_par (float, optional): Fraction of solar radiation (assuming
            black-body spectrum of 5800 K) that is photosynthetically
            active (wavelengths 400–700 nm).
        eta_photon (float, optional): Average number of photons per
            photosynthetically activate unit of radiation (in
            µmol s−1 W−1).
        method_solar_position (str, optional): Method that should be used
            by pvlib to determine the solar position.
        method_airmass (str, optional): Model that should be used by
            pvlib to determine the relative air mass.
        method_irradiance (str, optional): Model that should be used by
            pvlib to determine the solar irradiance.

    """

    def __init__(self, latitude, longitude, time, altitude=None,
                 pressure=None, temperature=12.0, eta_par=0.368,
                 eta_photon=4.56, method_solar_position='nrel_numpy',
                 method_airmass='kastenyoung1989',
                 method_irradiance='ineichen'):
        import pvlib
        self.pvlib = pvlib
        if pressure is None and altitude is None:
            pressure = 101325.0
            altitude = 0.0
        self.latitude = parse_quantity(latitude, 'degrees')
        self.longitude = parse_quantity(longitude, 'degrees')
        self.time = time
        self.altitude = parse_quantity(altitude, 'meters')
        self.pressure = parse_quantity(pressure, 'Pa')
        self.temperature = parse_quantity(temperature, 'degC')
        self.eta_par = eta_par
        self.eta_photon = parse_quantity(eta_photon, 'µmol s-1 W-1')
        self.method_solar_position = method_solar_position
        self.method_airmass = method_airmass
        self.method_irradiance = method_irradiance
        if self.pressure is None:
            self.pressure = parse_quantity(pvlib.atmosphere.alt2pres(
                self.altitude.value), 'Pa')
        if self.altitude is None:
            self.altitude = parse_quantity(pvlib.atmosphere.pres2alt(
                self.pressure.value), 'meters')
        self.location = pvlib.location.Location(
            self.latitude.value, self.longitude.value,
            altitude=self.altitude.value,
            tz=str(self.time.tzinfo),
        )
        self.time_pv = pvlib.tools._datetimelike_scalar_to_datetimeindex(
            self.time)
        self._solar_position = None
        self._irradiance = None

    @property
    def solar_position(self):
        r"""dict: Solar position information."""
        if self._solar_position is None:
            self._solar_position = self.location.get_solarposition(
                self.time_pv, pressure=self.pressure.value,
                temperature=self.temperature.value,
                method=self.method_solar_position,
            )
        return self._solar_position

    @property
    def apparent_elevation(self):
        r"""units.Quantity: Apparent elevation of the sun."""
        return parse_quantity(
            self.solar_position["apparent_elevation"].iloc[0], 'degrees')

    @property
    def azimuth(self):
        r"""units.Quantity: Azimuth angle of the sun."""
        return parse_quantity(
            self.solar_position["azimuth"].iloc[0], 'degrees')

    def relative_direction(self, up, north):
        from hothouse.blaster import SunRayBlaster
        blaster = SunRayBlaster(
            latitude=self.latitude.value, longitude=self.longitude.value,
            date=self.time, solar_altitude=self.apparent_elevation,
            solar_azimuth=self.azimuth,
            zenith=up.astype('f4'), north=north.astype('f4'),
            ground=np.zeros((3,), 'f4'),
            # direct_ppfd=self.ppfd_direct,
            # diffuse_ppfd=self.ppfd_diffuse,
        )
        return -blaster.forward

    # @property
    # def airmass(self):
    #     r"""pandas.DataFrame: Relative and absolute air mass."""
    #     return self.location.get_airmass(
    #         solar_position=self.solar_position,
    #         method=self.method_airmass,
    #     )

    @property
    def relative_airmass(self):
        r"""float: Relative (not pressure-adjusted) airmass at sea
        level."""
        return self.pvlib.atmosphere.get_relative_airmass(
            self.solar_position['apparent_zenith'])

    @property
    def absolute_airmass(self):
        r"""float: Absolute (pressure-adjusted) airmass."""
        return self.pvlib.atmosphere.get_absolute_airmass(
            self.relative_airmass, self.pressure)

    @property
    def linke_turbidity(self):
        r"""float: Linke Turibidity for the time/location."""
        return self.pvlib.clearsky.lookup_linke_turbidity(
            self.time_pv, self.latitude.value, self.longitude.value)

    @property
    def dni_extra(self):
        r"""units.Quantity: Extraterrestrial radiation incident on a
        surface normal to the sun (in W/m**2)."""
        return parse_quantity(
            self.pvlib.irradiance.get_extra_radiation(self.time_pv),
            "W m-2")

    @property
    def irradiance(self):
        r"""pandas.DataFrame: Solar irradiance."""
        if self._irradiance is None:
            self._irradiance = self.location.get_clearsky(
                self.time, model=self.method_irradiance,
                solar_position=self.solar_position,
                dni_extra=self.dni_extra,
                linke_turbidity=self.linke_turbidity,
                airmass_absolute=self.absolute_airmass,
            )
        return self._irradiance

    @property
    def dni(self):
        r"""units.Quantity: Direct normal irradiance"""
        return self.irradiance['dni']

    @property
    def dhi(self):
        r"""units.Quantity: Diffuse horizontal irradiance"""
        return self.irradiance['dhi']

    @property
    def ghi(self):
        r"""units.Quantity: Global horizontal irradiance"""
        return self.irradiance['ghi']

    @property
    def ppfd_direct(self):
        r"""units.Quantity: Direct photosynthetic photon flux density"""
        return self.eta_par * self.eta_photon * self.dni

    @property
    def ppfd_diffuse(self):
        r"""units.Quantity: Diffuse photosynthetic photon flux density"""
        return self.eta_par * self.eta_photon * self.dhi


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

    def __init__(self, args, mesh, plantids=None):
        super(RayTracerBase, self).__init__()
        if args.mesh_generated:
            assert plantids is not None
        self.args = args
        self.mesh = mesh
        self.plantids_face = plantids
        self.verbose = self.args.verbose
        self.mesh_dict = self.mesh.as_array_dict()
        if isinstance(self.mesh, ObjDict):
            self.mesh_dict['face'] -= 1
        self.areas = np.array(self.mesh.areas)
        self.area_mask = (self.areas > self._area_min)
        # print(f'{np.logical_not(self.area_mask).sum()} '
        #       f'faces have areas of 0')
        if self.args.include_units:
            self.areas = parse_quantity(
                self.areas, self.args.mesh_units**2)
        if self.args.plantids_in_blue:
            self.plantids_vertex = (
                255 * self.mesh_dict['vertex_colors'][:, 2]
            ).astype('uint8')
        else:
            self.plantids_vertex = np.zeros(
                (self.mesh_dict['vertex'].shape[0], ), dtype=np.uint8
            )
        self.plants = {}
        if self.args.separate_plants:
            if self.plantids_face is not None:
                for plantid in np.unique(self.plantids_face):
                    if self.args.verbose:
                        print(f'Selecting plant w/ ID \"{plantid}\"')
                    self.plants[plantid] = self.select_faces(
                        self.mesh_dict,
                        (self.plantids_face == plantid),
                        continuous=True,
                    )
            else:
                for plantid in np.unique(self.plantids_vertex):
                    if self.args.verbose:
                        print(f'Selecting plant w/ ID \"{plantid}\"')
                    self.plants[plantid] = self.select_vertices(
                        self.mesh_dict,
                        (self.plantids_vertex == plantid),
                        continuous=True,
                    )
        else:
            self.plants[0] = self.mesh_dict
        for plantid in self.plants.keys():
            self.plants[plantid] = self.select_faces(
                self.plants[plantid], self.area_mask,
                dont_prune_vertices=True,
            )
        self.log(f'Creating scene with up = {self.up}, '
                 f'north = {self.north}, '
                 f'east = {self.east}, ground = {self.ground}')

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
            mesh_dict = mesh.as_array_dict()
            if isinstance(mesh, ObjDict):
                mesh_dict['face'] -= 1
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
        return self.mesh_dict['vertex'].min(axis=0)

    @cached_args_property
    def scene_maxs(self):
        r"""np.ndarray: Maximum scene vertices in each dimension."""
        return self.mesh_dict['vertex'].max(axis=0)

    @cached_args_property
    def scene_limits(self):
        r"""np.ndarray: Corners of a box containing the scene."""
        limits = np.vstack([self.scene_mins, self.scene_maxs])
        xx, yy, zz = np.meshgrid(limits[:, 0], limits[:, 1], limits[:, 2])
        return np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T

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
        if self.args.camera_up:
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
        if self.args.camera_direction:
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
        if self.args.camera_location:
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
            return self.resolution
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

    def update_time(self, time):
        r"""Update the time represented by the ray tracer.

        Args:
            time (datetime.datetime): New time for tracing.

        """
        self.args.time = time


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
        return SolarModel(
            self.args.latitude, self.args.longitude, self.args.time,
            altitude=self.args.altitude, pressure=self.args.pressure,
            temperature=self.args.temperature,
            # TODO: Allow additional parameters to be passed?
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
        if self.args.periodic_canopy == 'rays':
            kws.update(
                period=self.args.periodic_period.astype('f4'),
                periodic_direction=self.args.periodic_direction.astype('f4'),
                periodic_count=(self.args.periodic_canopy_count
                                * np.ones((3, ), 'i4')),
            )
        return self.scene.get_sun_blaster(
            self.args.latitude, self.args.longitude, self.args.time,
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
            self.args.latitude, self.args.longitude,
            self.args.time,
            solar_altitude=self.solar_model.apparent_elevation,
            solar_azimuth=self.solar_model.azimuth,
            nx=10, ny=10, multibounce=False,
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
        self.log(f'Running ray tracer to get {self.args.query} for '
                 f't = {self.args.time} with sun '
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
        elif self.args.query == 'areas':
            return self.areas
        elif self.args.query == 'plantids':
            if self.plantids_face is not None:
                return self.plantids_face
            return self.plantids_vertex[self.mesh_dict['face'][:, 0]]
        else:
            raise ValueError(f"Unsupported ray tracer query "
                             f"\"{self.args.query}\"")
        values = np.zeros((self.mesh_dict['face'].shape[0], ), np.float64)
        for k, v in component_values.items():
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

class RayTraceTask(GenerateTask):
    r"""Class for running a ray tracer on a 3D canopy."""

    _name = 'raytrace'
    _time_vars = ['time']
    _hour_defaults = {'time': 12}
    _ext = '.csv'
    _output_dir = os.path.join(utils._output_dir, 'traces')
    _arguments_suffix_ignore = [
        'mesh', 'query', 'plantids_in_blue', 'separate_plants',
        'locaton', 'time', 'doy', 'hour', 'year', 'timezone',
        'show_rays', 'output_generate', 'overwrite_generate',
        'overwrite_raytrace', 'highlight', 'output_traced_mesh',
        'overwrite_traced_mesh',
    ]
    _alternate_outputs_write_optional = ['traced_mesh']
    _alternate_outputs_write_required = []
    _convert_to_mesh_units = [
        'ground_height',
        'ray_width', 'ray_length', 'arrow_width',
    ]
    _convert_to_color_tuple = [
        'ray_color',
    ]
    _argument_sources = [LayoutTask]
    _arguments = [
        (('--mesh', ), {
            'type': str,
            'help': ('Path to a file containing the mesh to raytrace. '
                     'If not provided, one will be generated.'),
        }),
        (('--plantids', ), {
            'type': str,
            'help': ('Path to a file containing plant IDs for the faces '
                     'in the provided mesh.'),
        }),
        (('--raytracer', ), {
            'type': str, 'default': 'hothouse',
            'choices': list(get_class_registry().keys('raytracer')),
            'help': 'Name of the ray tracer that should be used.',
        }),
        (('--separate-plants', ), {
            'action': 'store_true',
            'help': ('Track plants as separate components. This requires '
                     'that \"--plantids-in-blue\" is also set.'),
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
        (('--output-traced-mesh', ), {
            'nargs': '?', 'const': True, 'default': False,
            'help': ('File where the mesh should be saved with faces '
                     'colored by a ray tracer result. If the flag is '
                     'passed without a file name, a file name will be '
                     'generated.'),
        }),
        (('--overwrite-traced-mesh', ), {
            'action': 'store_true',
            'help': ('Overwrite any existing traced mesh file if '
                     '"--output-traced-mesh" is passed'),
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
                     'mesh, \"areas\" uses the area of each face, and '
                     '\"plantids\" uses the IDs of the plant each face '
                     'belongs to'),
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
        (('--ray-color', ), {
            'type': parse_color, 'default': '1.0,0.0,0.0',
            'help': ('Color that should be used for rays when '
                     '"--show-rays" is passed. This should be 3 '
                     'comma separated RGB values expressed as floats in '
                     'the range [0, 1]'),
        }),
        (('--highlight', ), {
            'type': str, 'choices': ['min', 'max'],
            'help': ('Highlight the face with the \"min\" or \"max\" '
                     'query value in the resulting (only valid if '
                     '"--output-traced-mesh" is passed).'),
        }),
        (('--highlight-color', ), {
            'type': parse_color, 'default': '1.0,0.0,1.0',
            'help': ('Color to use for highlighted faces if '
                     '"--highlight" is passed.'),
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
        (('--output-generate', ), {
            'type': str,
            'help': ('Path to file where generated mesh should be saved '
                     'other than the default if \"mesh\" is not '
                     'provided.'),
        }),
        (('--overwrite-generate', ), {
            'action': 'store_true',
            'help': ('Overwrite any existing generated mesh if \"mesh\" '
                     'is not provided.'),
        }),
        (('--plantids-in-blue', ), {
            'action': 'store_true',
            'help': 'Plant IDs are stored in the blue color channel. ',
        }),
    ]
    _argument_modifications = {
        '--output': {
            'help': ('File where the flux values for each face in the '
                     'mesh should be saved'),
        },
        '--mesh-format': {
            'help': ('Format that provided \"mesh\" is in or the format '
                     'that the generate mesh should be in if \"mesh\" '
                     'is not provided. If \"--mesh-format\" is not '
                     'provided, the file extension will be used to '
                     'determine the format'),
        },
        '--mesh-units': {
            'help': ('Units that the provided \"mesh\" is in or '
                     'the units the generated mesh should be in if '
                     '\"mesh\" is not provided'),
        },
        '--canopy': {
            'default': 'unique',
        },
        '--time': {
            'default': '2024-06-17',
        },
    }

    @classmethod
    def _read_output(cls, args, name=None):
        r"""Load an output file produced by this task.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            object: Contents of the output file.

        """
        if name is None:
            name = cls._name
        outputfile = getattr(args, f'output_{name}')
        if name == 'traced_mesh':
            return utils.read_3D(outputfile, file_format=args.mesh_format,
                                 verbose=args.verbose,
                                 include_units=args.include_units)
        return utils.read_csv(outputfile, verbose=args.verbose)

    @classmethod
    def _write_output(cls, output, args, name=None):
        r"""Write to an output file.

        Args:
            output (object): Output object to write to file.
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        """
        if name is None:
            name = cls._name
        outputfile = getattr(args, f'output_{name}')
        if name == 'traced_mesh':
            return utils.write_3D(output, outputfile,
                                  file_format=args.mesh_format,
                                  verbose=args.verbose)
        if args.crop_class == 'all_split':
            return
        utils.write_csv(output, outputfile, verbose=args.verbose)

    @property
    def verbose(self):
        r"""bool: Turn on log messages."""
        return self.args.verbose

    @readonly_cached_args_property
    def raytracer(self):
        r"""RayTracerBase: Ray tracer."""
        print("Re-creating ray tracer", self.args.time)
        assert self.plantids is not None
        return get_class_registry().get(
            'raytracer', self.args.raytracer)(
                self.args, self.mesh, plantids=self.plantids)

    @cached_property
    def mesh(self):
        r"""ObjDict: Mesh that will be ray traced."""
        return utils.read_3D(self.args.mesh,
                             file_format=self.args.mesh_format,
                             verbose=self.args.verbose)

    @cached_property
    def plantids(self):
        r"""np.ndarray: Plant IDs for each face in the mesh."""
        if os.path.isfile(self.args.plantids):
            return utils.read_csv(
                self.args.plantids,
                verbose=self.args.verbose,
                select='plantids',
            )
        return None

    @classmethod
    def adjust_args(cls, args):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        args.include_units = True
        if not hasattr(args, 'mesh_generated'):
            args.mesh_generated = (args.mesh is None)
        if not args.mesh_generated:
            args.periodic_canopy = False
        if args.mesh_generated:
            GenerateTask.adjust_args(args)
            args.mesh = args.output_generate
            args.plantids = args.output_plantids
        return super(RayTraceTask, cls).adjust_args(args)

    @classmethod
    def output_dir(cls, args, name=None):
        r"""Determine the directory that should be used to generate
        an output file name.

        Args:

            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            str: Directory.

        """
        if name == 'traced_mesh':
            return os.path.join(utils._output_dir, 'traced_meshes')
        return super(RayTraceTask, cls).output_dir(args, name=name)

    @classmethod
    def output_base(cls, args, name=None):
        r"""Generate the base file name that should be used to generate
        an output file name.

        Args:

            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            str: File base.

        """
        if name == 'traced_mesh':
            return args.output_raytrace
        return args.mesh

    @classmethod
    def output_suffix(cls, args, name=None):
        r"""Generate the suffix containing information about parameters
        that should be added to generated output files.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            str: Suffix.

        """
        if name == 'traced_mesh':
            suffix = f'_{args.query}'
            if getattr(args, 'show_rays', False):
                suffix += '_rays'
            if isinstance(getattr(args, 'highlight', False), str):
                suffix += f'_highlight{args.highlight.title()}'
            return suffix
        suffix = ''
        if args.location:
            suffix += f"_{args.location}"
        else:
            return False
        suffix += cls.output_suffix_time(args)
        suffix += f'_{args.nrays}'
        if args.multibounce:
            suffix += '_multibounce'
        if args.any_direction:
            suffix += '_anydirection'
        if args.periodic_canopy:
            suffix += (f'_periodic{args.periodic_canopy_count}'
                       f'_{args.periodic_canopy}')
        return suffix

    @classmethod
    def output_ext(cls, args, name=None):
        r"""Determine the extension that should be used for output files.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            str: Output file extension.

        """
        if name == 'traced_mesh':
            return os.path.splitext(args.mesh)[-1]
        return super(RayTraceTask, cls).output_ext(args, name=name)

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
    def query_limits(cls, query_values):
        r"""Compute limits from the query values.

        Args:
            query_values (dict): Query values for each face.

        Returns:
            dict: Limits on each query.

        """
        out = {}
        for query in _query_options:
            out[query] = {}
            values = cls.extract_query(query_values, query)
            if isinstance(values, units.QuantityArray):
                values = values.value
            if (values == 0).all():
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
        return out

    # TODO: This should only be modified if the date changes
    @cached_property
    def color_limits_noon(self):
        r"""tuple: Min/max for the queried values at noon."""
        assert not (self.args.time_str.endswith('noon')
                    and self.args.crop_class == self.all_crop_classes[0])
        print('GETTING MIN/MAX FROM NOON', self.args.time_str,
              self.args.crop_class)
        self.clear_cached_properties(include=['mesh'])
        query_values = RayTraceTask.run_class(
            self, args_overwrite={
                'time': 'noon',
                'crop_class': self.all_crop_classes[0],
                'output_raytrace': True,
                'output_generate': True,
                'mesh': None,
            },
        )
        self.clear_cached_properties(include=['mesh'])
        return self.query_limits(query_values)

    @classmethod
    def _set_color_limits(cls, self, query_values, name=None):
        r"""Set the minimum and maximum for color mapping.

        Args:
            self (object): Task instance that is running.
            query_values (dict): Query values for each face.
            name (str, optional): Name for limits to set if different
                than raytrace.

        """
        if name is None:
            name = cls._name
        var_min = f'color_vmin_{name}'
        var_max = f'color_vmax_{name}'
        var_scaling = f'colormap_scaling_{name}'
        if ((getattr(self.args, var_min) is not None
             and getattr(self.args, var_max) is not None)):
            return
        if ((self.args.time_str.endswith('noon')
             and self.args.crop_class == self.all_crop_classes[0])):
            self.color_limits_noon = self.query_limits(query_values)
        vscaling = getattr(self.args, var_scaling)
        limits = self.color_limits_noon[self.args.query]
        if getattr(self.args, var_min) is None:
            setattr(self.args, var_min, limits[f'vmin_{vscaling}'])
        if getattr(self.args, var_max) is None:
            setattr(self.args, var_max, limits[f'vmax_{vscaling}'])
        self.log(f'LIMITS[{cls._name}, {name}]: '
                 f'{getattr(self.args, var_min)}, '
                 f'{getattr(self.args, var_max)}', force=True)

    @classmethod
    def _color_scene(cls, self, query_values):
        r"""Run the ray tracer on the selected geometry.

        Args:
            self (object): Task instance that is running.
            query_values (dict): Query values for each face.

        Returns:
            ObjDict: Generated mesh with ray traced colors.

        """
        if self.args.show_rays:
            mesh = cls.run_class(
                self,
                args_overwrite={'show_rays': False},
                properties_preserve=['raytracer'],
                return_alternate_output='traced_mesh',
            )
            mesh.append(
                generate_rays(self.raytracer.ray_origins,
                              self.raytracer.ray_directions,
                              ray_length=self.raytracer.ray_lengths,
                              geom_format=type(mesh),
                              ray_color=self.args.ray_color,
                              ray_width=self.args.ray_width.value,
                              arrow_width=self.args.arrow_width.value)
            )
            return mesh
        cls._set_color_limits(self, query_values)
        mesh = self.mesh
        face_values = cls.extract_query(query_values, self.args.query)
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
            highlight_color=self.args.highlight_color,
        )
        mesh.add_colors('vertex', vertex_colors)
        return mesh

    @classmethod
    def _check_for_plantids(cls, self, values):
        if not self.args.mesh_generated:
            return True
        plantids = values['plantids']
        plantids_unique = np.unique(plantids)
        return len(plantids_unique) == self.args.nrows * self.args.ncols

    @classmethod
    def _raytrace_scene(cls, self):
        r"""Run the ray tracer on the selected geometry.

        Args:
            self (object): Task instance that is running.

        Returns:
            dict: Dictionary of ray tracer queries.

        """
        values = {}
        for k in _query_options:
            if k == 'flux':  # calculated from flux_density & areas
                continue
            self.cache_args(args_overwrite={'query': k},
                            properties_preserve=['raytracer'],
                            recursive=False)
            self.raytracer.args = self.args
            values[k] = self.raytracer.raytrace()
            self.restore_args()
            self.raytracer.args = self.args
        if self.args.output_traced_mesh:
            mesh = cls._color_scene(self, values)
            self.add_alternate_output('traced_mesh', mesh)
        return values

    @classmethod
    def raytrace_totals(cls, self, times=None, per_plant=False,
                        **kwargs):
        r"""Run the ray tracer on the selected geometry and compute the
        totals for each plant in the scene.

        Args:
            self (object): Task instance that is running.
            times (list, optional): Set of times to get values for. If
                not provided, only the current time will be used.
            per_plant (bool, optional): If True, the query values should
                also be totaled for each plant.
            **kwargs: Additional keyword arguments are passed to
                run_class.

        Returns:
            dict: Dictionary of ray tracer query totals.

        """
        if times is not None:
            kwargs.setdefault('args_overwrite', {})
            out = None
            for time in times:
                kwargs['args_overwrite']['time'] = time
                iout = cls.raytrace_totals(self, per_plant=per_plant,
                                           **kwargs)
                if out is None:
                    out = {k: {i: [] for i in ids.keys()}
                           for k, ids in iout.items()}
                for k, ids in iout.items():
                    for i, v in ids.items():
                        out[k][i].append(v)
            return out
        values = RayTraceTask.run_class(self, **kwargs)
        values['flux'] = cls.extract_query(values, 'flux')

        def sum(x):
            if isinstance(x, units.QuantityArray):
                return units.Quantity(x.value.sum(), x.units)
            return x.sum()

        out = {k: {'total': sum(values[k])} for k in _query_options}
        if per_plant:
            plantids = values['plantids']
            plantids_unique = np.unique(plantids)
            for i in plantids_unique:
                idx = (plantids == i)
                for k in _query_options:
                    out[k][i] = sum(values[k][idx])
        return out

    @classmethod
    def _run(cls, self, **kwargs):
        r"""Run the process associated with this subparser."""
        if self.args.mesh_generated and not os.path.isfile(self.args.mesh):
            if self.args.crop_class == 'all_split':
                meshes = []
                for crop_class in self.all_crop_classes:
                    self.clear_cached_properties(include=['mesh'])
                    cls.run_class(
                        self, args_overwrite={
                            'crop_class': crop_class,
                            'output_raytrace': True,
                            'output_generate': True,
                            'mesh': None,
                        },
                        args_preserve=[
                            'color_vmin_raytrace',
                            'color_vmax_raytrace',
                        ],
                        dont_load_existing=True,
                        dont_reset_alternate_output=True,
                        recursive=False,
                    )
                    if self.args.output_traced_mesh:
                        meshes.append(
                            self.pop_alternate_output('traced_mesh')
                        )
                if self.args.output_traced_mesh:
                    mesh = meshes[0]
                    x = 0.0 * self.args.x
                    y = 0.0 * self.args.y
                    plantid = 0
                    for imesh in meshes[1:]:
                        mesh.append(self.shift_mesh(
                            imesh, x, y, plantid=plantid,
                        ))
                        x += self.args.row_spacing * (self.args.nrows + 2)
                        plantid += (self.args.nrows * self.args.ncols)
                    self.add_alternate_output('traced_mesh', mesh)
                return None
        self.mesh = GenerateTask.run_class(
            self,
            args_preserve=['output_generate', 'output_plantids'],
        )
        self.args.mesh = self.args.output_generate
        self.args.plantids = self.args.output_plantids
        return cls._raytrace_scene(self, **kwargs)


class RenderTask(RayTraceTask):
    r"""Class for rendering a 3D canopy."""

    _name = 'render'
    _ext = '.png'
    _output_dir = os.path.join(utils._output_dir, 'render')
    _arguments_suffix_ignore = [
        'camera_direction', 'output_raytrace', 'overwrite_raytrace',
        'overwrite_render',
    ]
    _alternate_outputs_write_required = []
    _alternate_outputs_write_optional = []
    _convert_to_mesh_units = [
        'image_width', 'image_height',
    ]
    _convert_to_color_tuple = [
        'background',
    ]
    _arguments = [
        (('--camera-type', ), {
            'type': str,
            'choices': ['projection', 'orthographic'],  # 'spherical'],
            'default': 'projection',
            'help': ('Type of camera that should be used to render the '
                     'scene'),
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
                     'scene.'),
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
        (('--background', ), {
            'type': parse_color, 'default': 'transparent',
            'help': ('Background that should be used for the scene.'),
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
        (('--output-raytrace', ), {
            'action': 'store_false',
            'help': ('Output the raytraced mesh used to render the '
                     'scene'),
        }),
        (('--overwrite-raytrace', ), {
            'action': 'store_true',
            'help': ('Overwrite the output files for the raytraced mesh '
                     'used to render the scene'),
        }),
    ]
    _excluded_arguments = [
        '--show-rays', '--ray-color', '--ray-width', '--ray-length',
        '--arrow-width', '--highlight', '--highlight-color',
    ]
    _argument_modifications = {
        '--output': {
            'help': 'File where the rendered image should be saved',
        },
        '--colormap': {
            'default': 'viridis',
        },
        '--colormap-scaling': {
            'default': 'log',
        },
    }

    @classmethod
    def _read_output(cls, args, name=None):
        r"""Load an output file produced by this task.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            object: Contents of the output file.

        """
        outputfile = getattr(args, f'output_{cls._name}')
        return utils.read_png(outputfile, verbose=args.verbose)

    @classmethod
    def _write_output(cls, output, args, name=None):
        r"""Write to an output file.

        Args:
            output (object): Output object to write to file.
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        """
        outputfile = getattr(args, f'output_{cls._name}')
        utils.write_png(output, outputfile, verbose=args.verbose)

    @classmethod
    def adjust_args(cls, args, **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        args.show_rays = False
        if ((args.camera_direction is None
             and args.camera_location is None)):
            args.camera_direction = 'downnortheast'
        super(RenderTask, cls).adjust_args(args, **kwargs)

    @classmethod
    def output_suffix(cls, args, name=None, **kwargs):
        r"""Generate the suffix containing information about parameters
        that should be added to generated output files.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            str: Suffix.

        """
        suffix = super(RenderTask, cls).output_suffix(
            args, name=name, **kwargs)
        suffix += f'_{args.query}'
        if isinstance(args.camera_direction, str):
            suffix += f'_{args.camera_direction}'
        else:
            return False
        if args.camera_type != 'projection':
            suffix += f'_{args.camera_type}'
        background_str = None
        if isinstance(args.background, str):
            background_str = background_str
        elif getattr(args, 'background_str', None):
            background_str = args.background_str
        elif args.background:
            return False
        if background_str != 'transparent':
            suffix += f'_{background_str}'
        return suffix

    @classmethod
    def _render_scene(cls, self):
        r"""Render the scene using a ray tracer.

        Args:
            self (object): Task instance that is running.

        Returns:
            np.ndarray: Pixel color data.

        """
        query_values = RayTraceTask.run_class(
            self, args_overwrite={'query': None},
            properties_preserve=['raytracer'],
        )
        RayTraceTask._set_color_limits(self, query_values, name='render')
        face_values = RayTraceTask.extract_query(
            query_values, self.args.query)
        if isinstance(face_values, units.QuantityArray):
            face_values = np.array(face_values)
        pixel_values = self.raytracer.render(face_values)
        if isinstance(pixel_values, units.QuantityArray):
            pixel_values = np.array(pixel_values)
        self.add_alternate_output('pixel_values', pixel_values)
        pixel_values = (pixel_values.T)[::-1, :]
        image = utils.apply_color_map(
            pixel_values,
            color_map=self.args.colormap_render,
            vmin=self.args.color_vmin_render,
            vmax=self.args.color_vmax_render,
            scaling=self.args.colormap_scaling_render,
            highlight=(pixel_values < 0),
            highlight_color=self.args.background,
            include_alpha=(len(self.args.background) == 4)
        )
        return image

    @classmethod
    def _run(cls, self, **kwargs):
        r"""Run the process associated with this subparser."""
        if self.args.mesh_generated and not os.path.isfile(self.args.mesh):
            if self.args.crop_class == 'all_split':
                images = []
                for crop_class in self.all_crop_classes:
                    self.clear_cached_properties(include=['mesh'])
                    images.append(cls.run_class(
                        self, args_overwrite={
                            'crop_class': crop_class,
                            'output_render': True,
                            'output_generate': True,
                            'mesh': None,
                        },
                        args_preserve=[
                            'color_vmin_render',
                            'color_vmax_render',
                        ],
                        recursive=False,
                    ))
                # TODO: Verify that this is correct axis
                image = np.concatenate(images, axis=1)
                return image
            self.mesh = GenerateTask.run_class(
                self,
                args_preserve=['output_generate', 'output_plantids'],
            )
            self.args.mesh = self.args.output_generate
            self.args.plantids = self.args.output_plantids
        return cls._render_scene(self, **kwargs)


class AnimateTask(TemporalTaskBase(RenderTask, step_alias='frame')):
    r"""Class for producing an animation."""

    _name = 'animate'
    _ext = None
    _output_dir = os.path.join(utils._output_dir, 'movies')
    _arguments_suffix_ignore = [
        'movie_format', 'output_render',
        'overwrite_render', 'output_totals', 'overwrite_totals',
    ]
    _alternate_outputs_write_required = []
    _alternate_outputs_write_optional = ['totals']
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
        (('--output-totals', ), {
            'nargs': '?', 'const': True, 'default': False,
            'help': ('Output a plot with the totals as a function '
                     'of time.'),
        }),
        (('--overwrite-totals', ), {
            'action': 'store_true',
            'help': ('Overwrite any existing plot of the totals as a '
                     'function of time.'),
        }),
        (('--inset-totals', ), {
            'action': 'store_true',
            'help': ('Inset a plot of the query total below the '
                     'image.'),

        }),
    ]
    _argument_modifications = {
        '--output': {
            'help': 'File where the generated animation should be saved',
        },
    }
    # _excluded_arguments = [
    #     '--time',
    # ]

    @classmethod
    def _read_output(cls, args, name=None):
        r"""Load an output file produced by this task.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            object: Contents of the output file.

        """
        # outputfile = getattr(args, f'output_{cls._name}')
        raise NotImplementedError

    @classmethod
    def _write_output(cls, output, args, name=None):
        r"""Write to an output file.

        Args:
            output (object): Output object to write to file.
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        """
        if name is None:
            name = cls._name
        outputfile = getattr(args, f'output_{name}')
        if name == 'totals':
            output.savefig(outputfile, dpi=300)
            return
        utils.write_movie(output, outputfile, frame_rate=args.frame_rate,
                          verbose=args.verbose)

    @classmethod
    def adjust_args(cls, args, **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        args.output_render = True
        super(AnimateTask, cls).adjust_args(args, **kwargs)
        for k in ['colormap', 'color_vmin', 'color_vmax',
                  'colormap_scaling']:
            setattr(args, f'{k}_render', getattr(args, f'{k}_animate'))

    @classmethod
    def output_suffix(cls, args, name=None, **kwargs):
        r"""Generate the suffix containing information about parameters
        that should be added to generated output files.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            str: Suffix.

        """
        suffix = super(AnimateTask, cls).output_suffix(
            args, name=name, **kwargs)
        if args.inset_totals:
            suffix += '_totals'
        # if args.per_plant_totals:
        #     suffix += '_per_plant'
        if args.frame_rate != 1:
            suffix += f'_{args.frame_rate}fps'
        return suffix

    @classmethod
    def output_ext(cls, args, name=None):
        r"""Determine the extension that should be used for output files.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            str: Output file extension.

        """
        return f'.{args.movie_format}'

    @cached_property
    def figure_totals(self):
        r"""Figure containing the query totals."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import matplotlib.units as munits
        converter = mdates.ConciseDateConverter()
        munits.registry[datetime] = converter
        times = self.times
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if self.args.crop_class == 'all_split':
            crop_classes = self.all_crop_classes
        else:
            crop_classes = [self.args.crop_class]
        ylabel = None
        xlim = None
        ylim = None
        linestyles = ['-', ':']
        colors = ['b', 'o']
        for iclass, crop_class in enumerate(crop_classes):
            totals = RayTraceTask.raytrace_totals(
                self, times=times,
                per_plant=self.args.per_plant_totals,
                args_overwrite={
                    'crop_class': crop_class,
                    'output_render': True,
                    'output_generate': True,
                    'mesh': None,
                },
            )[self.args.query]
            if ylabel is None:
                ylabel = self.args.query.title()  # TODO: Add units
                if isinstance(totals['total'], units.QuantityArray):
                    ylabel += f" ({totals['total'].units})"
                elif isinstance(totals['total'][0], units.Quantity):
                    ylabel += f" ({totals['total'][0].units})"
            if self.args.per_plant_totals:
                lines_int = {}
                lines_ext = {}
                # TODO: Split plants by crop class when they are
                # combined in the same mesh
                for k, v in totals.items():
                    loc = None
                    label = None
                    if k == 'total':
                        continue
                    if self.isExteriorPlant(k):
                        loc = 1
                        lines_ext[k] = v
                        if (len(lines_ext) == 1):
                            label = f'Exterior plants ({crop_class})'
                    else:
                        loc = 0
                        lines_int[k] = v
                        if (len(lines_int) == 1):
                            label = f'Interior plants ({crop_class})'
                    color = colors[iclass]
                    style = linestyles[loc]
                    ax.plot(times, v, label=label, color=color,
                            linestyle=style)
            else:
                ax.plot(times, totals['total'],
                        label=f'total ({crop_class})')
        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend()
        return fig

    @property
    def time_marker(self):
        r"""matplotlib.lines.line2D: Vertical line marking the time."""
        ax = self.figure_totals.get_axes()[0]
        return ax.axvline(x=self.start_time,
                          color=(1, 1, 1), alpha=0.5,
                          linewidth=10)

    @cached_property
    def inset_figure(self):
        r"""LightTask: Light instance."""
        if not self.args.inset_totals:
            return None
        light_task = LightTask(args=self.args)
        frame = self.args.output_render
        old_data = utils.read_png(frame, verbose=self.args.verbose)
        print('OLD_IMAGE', old_data.shape)
        pdb.set_trace()
        width_px = old_data.shape[0]
        height_px = int(0.2 * width_px)
        dpi = light_task.figure.get_dpi()
        light_task.figure.set_size_inches(width_px / dpi,
                                          height_px / dpi)
        return light_task

    def add_totals_to_frame(self, time, frame=None):
        r"""Add a plot of query totals to a frame.

        Args:
            time (datetime.datetime): Time that frame is associated with.
            frame (str, optional): Frame containing the rendered scene
                that the plot should be added to. If not provided, the
                most recent value of self.args.output_render is used.

        Returns:
            str: Updated frame with the plot added.

        """
        update_args = False
        if frame is None:
            update_args = True
            frame = self.args.output_render
        if not self.inset_figure:
            return frame
        old_data = utils.read_png(frame, verbose=self.args.verbose)
        frame_new = '_totals'.join(os.path.splitext(frame))
        self.inset_figure.mark_time(time)
        data = self.inset_figure.raw_figure_data
        print('NEW_IMAGE', data.shape)
        pdb.set_trace()
        data_new = np.concatenate([old_data, data])
        print('CONCAT', data_new)
        utils.write_png(data_new, frame_new, verbose=self.args.verbose)
        if update_args:
            self.args.output_render = frame_new
        return frame_new

    @classmethod
    def _run_step(cls, self, time, **kwargs):
        kwargs.setdefault('args_preserve', [])
        kwargs['args_preserve'] += [
            'color_vmin_render', 'color_vmax_render',
        ]
        super(AnimateTask, cls)._run_step(self, time, **kwargs)
        self.add_totals_to_frame(time)
        return self.args.output_render


class LightTask(TemporalTaskBase(RayTraceTask)):
    r"""Class for plotting the flux on a geometry as a function of time."""
    _name = 'light'
    _ext = '.png'
    _output_dir = os.path.join(utils._output_dir, 'light')
    _alternate_outputs_write_required = []
    _alternate_outputs_write_optional = []
    _arguments = [
        (('--per-plant', ), {
            'action': 'store_true',
            'help': ('Plot the totals on a per-plant basis'),
        }),
    ]
    _argument_modifications = {
        '--output': {
            'help': 'File where the generated plot should be saved',
        },
    }

    @classmethod
    def _read_output(cls, args, name=None):
        r"""Load an output file produced by this task.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            object: Contents of the output file.

        """
        # if name is None:
        #     name = cls._name
        # outputfile = getattr(args, f'output_{name}')
        # return utils.read_png(outputfile, verbose=args.verbose)
        raise NotImplementedError

    @classmethod
    def _write_output(cls, output, args, name=None):
        r"""Write to an output file.

        Args:
            output (object): Output object to write to file.
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        """
        if name is None:
            name = cls._name
        outputfile = getattr(args, f'output_{name}')
        output.savefig(outputfile, dpi=300)

    @classmethod
    def output_suffix(cls, args, name=None, **kwargs):
        r"""Generate the suffix containing information about parameters
        that should be added to generated output files.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            str: Suffix.

        """
        suffix = super(LightTask, cls).output_suffix(
            args, name=name, **kwargs)
        if args.per_plant:
            suffix += '_per_plant'
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

    def plot_data(self, totals, crop_class=None, iclass=0):
        if crop_class is None:
            crop_class = self.args.crop_class
            iclass = 0
        totals = totals[self.args.query]
        first = getattr(self, 'first', True)
        times = self.times
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
                    label = f'{locStr.title()} plants ({crop_class})'
                color = colors[iclass]
                style = linestyles[loc]
                ax.plot(times, v, label=label, color=color,
                        linestyle=style)
            # print(f'INTERIOR PLANTS [{crop_class}]: '
            #       f'{len(lines["interior"])}/{len(totals) - 1}')
            # print(f'EXTERIOR PLANTS [{crop_class}]: '
            #       f'{len(lines["exterior"])}/{len(totals) - 1}')
        else:
            color = colors[iclass]
            style = linestyles[0]
            ax.plot(times, totals['total'], color=color,
                    label=f'total ({crop_class})',
                    linestyle=style)

    @classmethod
    def _run_step(cls, self, time, crop_class=None, **kwargs):
        if crop_class is not None:
            kwargs.setdefault('args_overwrite', {})
            kwargs['args_overwrite'].update(
                crop_class=crop_class,
                mesh=None,
                output_generate=True,
                output_raytrace=True,
            )
        if self.args.per_plant:
            kwargs.setdefault('args_overwrite', {})
            kwargs['args_overwrite'].update(
                output_plantids=True,
            )
            kwargs.setdefault('args_preserve', [])
            kwargs['args_preserve'].append('output_plantids')
        values = super(LightTask, cls)._run_step(self, time, **kwargs)
        if not RayTraceTask._check_for_plantids(self, values):
            kwargs.setdefault('args_overwrite', {})
            kwargs['args_overwrite'].update(
                overwrite_raytrace=True,
            )
            print("REGENERATING RAY TRACE WITHOUT PLANTIDS")
            values = super(LightTask, cls)._run_step(self, time, **kwargs)
            assert RayTraceTask._check_for_plantids(self, values)
        values['flux'] = cls.extract_query(values, 'flux')

        def sum(x):
            if isinstance(x, units.QuantityArray):
                return units.Quantity(x.value.sum(), x.units)
            return x.sum()

        out = {k: {'total': sum(values[k])} for k in _query_options}
        if self.args.per_plant:
            plantids = values['plantids']
            plantids_unique = np.unique(plantids)
            if self.args.mesh_generated:
                assert (len(plantids_unique)
                        == self.args.nrows * self.args.ncols)
            for i in plantids_unique:
                idx = (plantids == i)
                for k in _query_options:
                    out[k][i] = sum(values[k][idx])
        return out

    @classmethod
    def _run(cls, self, crop_class=None, iclass=None, **kwargs):
        r"""Run the process associated with this subparser."""
        if self.args.crop_class == 'all_split' and crop_class is None:
            # out = {}
            for iclass, crop_class in enumerate(self.all_crop_classes):
                cls._run(self, crop_class=crop_class, iclass=iclass,
                         **kwargs)
                # out[crop_class] = cls._run(
                #     self, crop_class=crop_class, **kwargs
                # )
            # return out
            self.axes.legend()
            return self.figure
        out = None
        for time in self.times:
            iout = cls._run_step(self, time, crop_class=crop_class,
                                 **kwargs)
            if out is None:
                out = {k: {i: [] for i in ids.keys()}
                       for k, ids in iout.items()}
            for k, ids in iout.items():
                for i, v in ids.items():
                    out[k][i].append(v)
        self.plot_data(out, crop_class=crop_class, iclass=iclass, **kwargs)
        if crop_class is None:
            self.axes.legend()
        return self.figure
