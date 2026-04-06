import numpy as np
from canopy_factory.utils import (
    cached_args_property,
    parse_quantity,
)
from yggdrasil_rapidjson import units
from canopy_factory.raytrace.base import RayTracerBase
from hothouse.model import Model
from hothouse.scene import Scene, PeriodicScene


class HothouseRayTracer(RayTracerBase):
    r"""Raytrace using hothouse."""

    _name = 'hothouse'
    RTC_INVALID_GEOMETRY_ID = np.int32(-1)

    @cached_args_property
    def scene(self):
        r"""hothouse.scene.Scene: Scene containing geometry."""
        kws = {
            'ground': self.ground,
            'up': self.up,
            'north': self.north,
        }
        scene_cls = Scene
        if ((self.args.periodic_canopy == 'scene'
             or self.args.canopy == 'virtual_single')):
            scene_cls = PeriodicScene
            if self.args.canopy == 'virtual_single':
                virtual_count = self.args.virtual_canopy_count_array
                if self.args.periodic_canopy == 'scene':
                    virtual_count = (
                        virtual_count
                        * (2 * self.args.periodic_canopy_count_array + 1)
                    )
                kws.update(
                    period=self.args.virtual_period,
                    direction=self.args.virtual_direction,
                    count=virtual_count,
                    dont_reflect=True,
                )
            else:
                kws.update(
                    period=self.args.periodic_period,
                    direction=self.args.periodic_direction,
                    count=self.args.periodic_canopy_count_array,
                )
        out = scene_cls(**kws)
        self._plantid2geomID = {}
        self._geomID2primID = {}
        self._geomID2realID = {}
        self._currentIDs = {
            'geomID': 0,
            'primID': 0,
            'realID': 0,
        }

        def add_plant(plant_type, plantid, triangles, baseID=None,
                      shift=None):
            mesh_dict = self.plants[plantid]
            geomID = self._currentIDs['geomID']
            primID = self._currentIDs['primID']
            realID = self._currentIDs['realID']
            kws = dict(
                vertices=mesh_dict['vertex'],
                indices=mesh_dict['face'].astype('i4'),
                attributes=mesh_dict['vertex_colors'],
                triangles=triangles,
            )
            if shift is None:
                shift = np.zeros((3,), 'f8')
            else:
                for k in ['vertices', 'triangles']:
                    kws[k] = np.round(kws[k] + shift, decimals=6)
            plant = Model(**kws)
            if plant_type == 'real':
                self._plantid2geomID[plantid] = {
                    'real': geomID,
                    'virtual': [],
                    'periodic': {},
                }
            else:
                if plant_type == 'periodic':
                    assert baseID is not None
                    self._plantid2geomID[plantid][plant_type].setdefault(
                        baseID, [])
                    self._plantid2geomID[plantid][plant_type][
                        baseID].append(geomID)
                elif plant_type == 'virtual':
                    self._plantid2geomID[plantid][plant_type].append(geomID)
                else:
                    raise NotImplementedError(plant_type)
            self._geomID2primID[geomID] = range(
                primID, primID + triangles.shape[0])
            self._geomID2realID[geomID] = range(
                realID, realID + triangles.shape[0])
            self._currentIDs['geomID'] += 1
            self._currentIDs['primID'] += triangles.shape[0]
            if plant_type in ['real', 'virtual']:
                self._currentIDs['realID'] += triangles.shape[0]
            assert geomID == len(out.components)
            out.add_component(plant)
            if ((plant_type in ['real', 'virtual']
                 and self.args.periodic_canopy == 'plants')):
                for ishift in self.periodic_shifts:
                    add_plant('periodic', plantid, triangles,
                              baseID=geomID, shift=(shift + ishift))
            if ((plant_type == 'real'
                 and not (self.args.canopy == 'virtual_single'
                          and isinstance(out, PeriodicScene)))):
                for ishift in self.virtual_shifts:
                    add_plant('virtual', plantid, triangles,
                              shift=(shift + ishift))

        for plantid, triangles in self.plant_triangles:
            add_plant('real', plantid, triangles)
        return out

    @cached_args_property
    def coverage_blaster(self):
        r"""hothouse.blaster.OrthographicRayBlaster: Blaster for coverage."""
        from hothouse.blaster import OrthographicRayBlaster
        out = OrthographicRayBlaster(
            center=self.zenith_origin,
            forward=-self.up,
            up=self.args.axis_cols,
            width=self.args.plot_width,
            height=self.args.plot_length,
            nx=self.args.nrays,
            ny=self.args.nrays,
        )
        return out

    @cached_args_property
    def camera_blaster(self):
        r"""hothouse.blaster.RayBlaster: Blaster for camera."""
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
            center=self.image_center,
            forward=self.camera_direction,
            up=self.camera_up,
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
        r"""hothouse.blaster.SphericalRayBlaster: Blaster for a point
        source."""
        from hothouse.blaster import SphericalRayBlaster
        assert self.args.periodic_canopy != 'rays'
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
            **kws
        )

    def get_solar_blaster(self, **kwargs):
        r"""Create a hothouse SunRayBlaster.

        Args:
            **kwargs: All keyword arguments are passed to the
                SunRayBlaster constructor after supplementing missing
                arguments with scene properties.

        Returns:
            SunRayBlaster: Blaster instance.

        """
        from hothouse.blaster import SunRayBlaster
        kws = dict(
            latitude=self.solar_model.latitude.value,
            longitude=self.solar_model.longitude.value,
            date=self.solar_model.time,
            intensity_density=self.solar_model.ppfd_direct.value[0],
            diffuse_intensity=self.solar_model.ppfd_diffuse.value[0],
            ground=self.ground,
            north=self.north,
            zenith=self.zenith,
            scene_limits=self.virtual_scene_model.limits,
            solar_altitude=self.solar_model.apparent_elevation,
            solar_azimuth=self.solar_model.azimuth,
            nx=self.args.nrays, ny=self.args.nrays,
        )
        if self.args.periodic_canopy == 'rays':
            kws.update(
                period=self.args.periodic_period,
                periodic_direction=self.args.periodic_direction,
                periodic_count=self.args.periodic_canopy_count_array,
            )
        for k, v in kws.items():
            kwargs.setdefault(k, v)
        return SunRayBlaster(**kwargs)

    @cached_args_property
    def solar_blaster(self):
        r"""hothouse.blaster.SolarBlaster: Blaster for sun."""
        # TODO: Add units to parser
        self.log(f"Total PPFD"
                 f"\n   direct = {self.solar_model.ppfd_direct}"
                 f"\n   diffuse = {self.solar_model.ppfd_diffuse}",
                 force=True)
        blaster = self.get_solar_blaster()
        # blaster.intensity = (
        #     self.solar_model.ppfd_direct.value[0]
        #     * blaster.width * blaster.height
        # )
        return blaster

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
        rb = self.get_solar_blaster(nx=10, ny=10)
        ray_origins = rb.origins
        ray_directions = rb.directions
        ray_lengths = self.args.ray_length.value
        if ray_lengths < 0:
            ray_length0 = -ray_lengths
            ray_lengths = rb.compute_distance(self.scene)
            idx_max = (ray_lengths >= max(ray_lengths))
            ray_lengths[idx_max] = ray_length0
        return (ray_origins, ray_directions, ray_lengths)

    def raytrace_sample(self, hits, values, value_miss=np.nan,
                        method='accumulate'):
        r"""Sample a set of values for each face in the mesh based on
        the faces hit by the ray tracer.

        Args:
            hits (dict): Raytracer result.
            values (np.ndarray): Array of values at each face.
            value_miss (np.float64, optional): Value to use when a ray
                did not hit a face.
            method (str, optional): How the values should be sampled.

        Returns:
            np.ndarray: Result for each ray.

        """
        if method in ['multiply']:
            out = np.ones(hits['geomID'].shape, "f8")
        else:
            out = np.zeros(hits['geomID'].shape, "f8")
        if isinstance(values, units.QuantityArray):
            out = units.QuantityArray(out, values.units)
            value_miss = parse_quantity(value_miss, values.units)
        any_hits = (hits["primID"] != self.RTC_INVALID_GEOMETRY_ID)

        def assign_geomID(geomID, v):
            idx_hits = np.logical_and(hits["geomID"] == geomID, any_hits)
            idx_scene = hits["primID"][idx_hits]
            if method == 'accumulate':
                out[idx_hits] += v[idx_scene]
            elif method == 'multiply':
                out[idx_hits] *= v[idx_scene]
            else:
                raise NotImplementedError(method)

        for plantid, mesh_dict in self.plants.items():
            for ivirt, geomID in enumerate(self.plantid2geomIDs(plantid)):
                if self.args.canopy == 'virtual':
                    shift = (ivirt * self.mesh_dict['face'].shape[0])
                else:
                    shift = 0
                idx_src = self.finalize_face_index(
                    mesh_dict.get('idx', None),
                    self.mesh_dict['face'].shape[0],
                    shift=shift,
                )
                v = values[idx_src]
                assign_geomID(geomID, v)
                for periodic_geomID in self._plantid2geomID[plantid][
                        'periodic'].get(geomID, []):
                    assign_geomID(periodic_geomID, v)
        out[hits["primID"] == self.RTC_INVALID_GEOMETRY_ID] = value_miss
        return out

    def raytrace(self, query):
        r"""Run the ray tracer and get values for each face.

        Args:
            query (str): Type of raytrace operation that should be
                performed.

        Returns:
            np.ndarray: Ray tracer results for each face.

        """
        if self.geometryids and query in self.geometryids:
            out = self.geometryids[query]
            if self.args.canopy == 'virtual':
                out = [out]
                if query == 'plantids':
                    maxid = out[0].max() + 1
                    for ivert in range(1, self.nvirtual + 1):
                        out.append(out[0] + ivert * maxid)
                elif query in ['componentids', 'areas', 'height']:
                    for ivert in range(1, self.nvirtual + 1):
                        out.append(out[0])
                else:
                    raise NotImplementedError(query)
                out = np.hstack(out)
            return out
        nfaces = self.mesh_dict['face'].shape[0]
        if self.args.canopy == 'virtual':
            nfaces *= (self.nvirtual + 1)
        value_units = None
        if self.args.include_units:
            if query == 'flux_density':
                value_units = self.solar_model.ppfd_direct.units
        values = np.zeros((nfaces, ), np.float64)
        if values.shape[0] == 0:
            if value_units:
                values = parse_quantity(values, value_units)
            return values
        self.log(f'Running ray tracer to get {query} for '
                 f't = {self.args.time.time}, age = {self.args.time.age} '
                 f'({self.args.age.value}) with '
                 f'light direction: {self.solar_blaster.forward}',
                 border=True, force=True)
        component_values = None
        if query == 'flux_density':
            component_values = self.scene.compute_flux_density(
                self.solar_blaster,
                any_direction=self.args.any_direction,
                multibounce=self.args.multibounce,
            )
        elif query == 'hits':
            component_values = self.scene.compute_hit_count(
                self.solar_blaster, multibounce=self.args.multibounce)
        else:
            raise ValueError(f"Unsupported ray tracer query "
                             f"\"{query}\"")
        for plantid, mesh_dict in self.plants.items():
            idx = mesh_dict.get('idx', None)
            geomIDs = self.plantid2geomIDs(
                plantid, exclude_virtual=(self.args.canopy != 'virtual'))
            for ivirt, geomID in enumerate(geomIDs):
                v = component_values[geomID]
                idx_dst = self.finalize_face_index(
                    idx, self.mesh_dict['face'].shape[0],
                    shift=(ivirt * self.mesh_dict['face'].shape[0]),
                )
                assert idx_dst.shape == v.shape
                values[idx_dst] = v
        if value_units:
            values = parse_quantity(values, value_units)
        return values

    def plantid2geomIDs(self, plantid, exclude_virtual=False):
        r"""Get the scene geometry IDs for a given plant ID including both
        real and virtual plant geometries.

        Args:
            plantid (int): Plant ID to get geometry IDs for.
            exclude_virtual (bool, optional): If True, don't include
                virtual plant geometries.

        Returns:
            list: Geometry IDs.

        """
        out = [self._plantid2geomID[plantid]['real']]
        if not exclude_virtual:
            out += self._plantid2geomID[plantid]['virtual']
        return out

    def coverage(self):
        r"""Compute the scene coverage.

        Returns:
            float: Percent of rays that hit the plants.

        """
        prev = False
        try:
            if hasattr(self.scene, 'buffer_as_primary'):
                prev = self.scene.buffer_as_primary
                self.scene.buffer_as_primary = True
            hits = self.coverage_blaster.compute_count(self.scene)
        finally:
            if hasattr(self.scene, 'buffer_as_primary'):
                self.scene.buffer_as_primary = prev
        return (
            sum(hits["primID"] != self.RTC_INVALID_GEOMETRY_ID)
            / len(hits["primID"])
        )

    def render(self, values, value_miss=-1.0, blaster=None):
        r"""Image the scene.

        Args:
            values (np.ndarray): Values on each face that should be used
                when imaging the scene.
            value_miss (float, optional): Value that should be used for
                pixels that do not hit anything.

        Returns:
            np.ndarray: Ray tracer results for each pixel.

        """
        if blaster is None:
            blaster = self.camera_blaster
        prev = False
        try:
            if hasattr(self.scene, 'buffer_as_primary'):
                prev = self.scene.buffer_as_primary
                self.scene.buffer_as_primary = True
            # TODO: Pass multibounce to allow reflection etc.
            hits = blaster.compute_count(self.scene)
        finally:
            if hasattr(self.scene, 'buffer_as_primary'):
                self.scene.buffer_as_primary = prev
        out = self.raytrace_sample(
            hits, values, value_miss=value_miss)
        return out.reshape((blaster.nx, blaster.ny))
