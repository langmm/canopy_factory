import numpy as np
from yggdrasil_rapidjson import units
from canopy_factory import arguments
from canopy_factory.utils import cached_args_property
from canopy_factory.cli import SubparserBase


class LightSourceBase(SubparserBase):
    r"""Base class for light sources."""

    _registry_key = 'light_source'
    _name = None
    _default = 'sun'
    _arguments = [
        arguments.CompositeArgumentDescription(
            'light', description=' of the light source',
        ),
    ]

    @property
    def geometry(self):
        r"""str: Light geometry."""
        if self._geometry is None:
            raise NotImplementedError
        return self._geometry

    @property
    def ppf_direct(self):
        r"""units.Quantity: Direct photosynthetic photon flux."""
        raise NotImplementedError

    @property
    def ppf_diffuse(self):
        r"""units.Quantity: Diffuse photosynthetic photon flux."""
        raise NotImplementedError


class SolarLightSource(LightSourceBase):
    r"""Solar light source."""

    _name = 'sun'
    _geometry = 'orthographic'
    _arguments = [LightSourceBase._arguments.copy(
        modifications={
            'eta_par': {
                'default': 0.368,
                'help': (
                    'Fraction of solar radiation (assuming black-body '
                    'spectrum of 5800 K) that is photosynthetically '
                    'active (wavelengths 400–700 nm).'
                ),
            },
            'eta_photon': {
                'default': 4.56,
                'help': (
                    'Average number of photons per photosynthetically '
                    'activate unit of radiation (in µmol s−1 W−1).'
                ),
            },
        },
    ),
        arguments.CompositeArgumentDescription(
            'location',
            description=' that the light should be modeled for',
            defaults={
                'location': 'Champaign',
            },
            suffix_param={'noteq': 'Champaign'},
        ),
        arguments.CompositeArgumentDescription(
            'time',
            description=' that the light should be modeled for',
            defaults={
                'hour': 'noon',
                'doy': 'summer_solstice',
            },
            suffix_param={},
        ),
        (('--method-solar-position', ), {
            'type': str, 'default': 'nrel_numpy',
            'choices': ['nrel_numpy'],
            'help': (
                'Method that should be used by pvlib to determine the '
                'relative air mass.'
            ),
        }),
        (('--method-airmass', ), {
            'type': str, 'default': 'kastenyoung1989',
            'choices': ['kastenyoung1989'],
            'help': (
                'Model that should be used by pvlib to determine the '
                'relative air mass.'
            ),
        }),
        (('--method-irradiance', ), {
            'type': str, 'default': 'ineichen',
            'choices': ['ineichen'],
            'help': (
                'Model that should be used by pvlib to determine the '
                'solar irradiance.'
            ),
        }),
    ]

    @classmethod
    def adjust_args(cls, args, skip=None):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            skip (list, optional): Arguments to skip.

        """
        print("HERE")
        # 1. Initialize location
        # 2. Transfer timezone from location to time
        # Initialize time first, then use solar model to set
        # irradiance
        for k in ['location', 'time']:
            cls._convert_composite(args, k, base=k)
        super(SolarLightSource, cls).adjust_args(args, skip=skip)
        for k in ['method_solar_position', 'method_solar_position']:
            setattr(args.time.solar_model, k, getattr(args, k))

    @cached_args_property
    def solar_model(self):
        r"""SolarModel: Model for the sun using pvlib."""
        kws = {
            k: getattr(self, k) for k in [
                'method_solar_position', 'method_airmass',
                'method_irradiance'
            ]
        }
        return self.location.create_solar_model(self.time.time, **kws)

    @property
    def ppfd_direct(self):
        r"""units.Quantity: Direct photosynthetic photon flux density"""
        return self.solar_model.ppfd_direct

    @property
    def ppfd_diffuse(self):
        r"""units.Quantity: Diffuse photosynthetic photon flux density"""
        return self.solar_model.ppfd_diffuse


class BulbTypeBase(SubparserBase):
    r"""Base class for different bulb types."""

    _registry_key = 'bulb_type'
    _name = None


class IncandescentBulb(BulbTypeBase):
    r"""Incandescent bulb."""

    _name = 'incandescent'


class FluorescentBulb(BulbTypeBase):
    r"""Fluorescent bulb."""

    _name = 'fluorescent'


class LEDBulb(BulbTypeBase):
    r"""LED bulb."""

    _name = 'LED'


class BulbShapeBase(SubparserBase):
    r"""Base class for different bulb shapes."""

    _registry_key = 'bulb_shape'
    _name = None
    _default = 'sphere'
    _arguments = [
        (('--bulb-width', ), {
            'units': 'cm',
            'help': 'Width of the bulb (in cm)',
        }),
    ]

    @property
    def surface_area(self):
        r"""units.Quantity: The bulb surface area."""
        raise NotImplementedError

    @property
    def volume(self):
        r"""units.Quantity: The bulb volume."""
        raise NotImplementedError


class TubeBulb(BulbShapeBase):
    r"""Tube bulb."""

    _name = 'tube'
    _arguments = BulbShapeBase._arguments.copy(
        modifications={
            'bulb_width': {'default': units.Quantity(2.5, 'cm')},
        },
    ) + [
        (('--bulb-length', ), {
            'units': 'cm',
            'default': units.Quantity(121.92, 'cm'),
            'help': 'Length of the bulb (in cm)',
        }),
    ]

    @property
    def surface_area(self):
        r"""units.Quantity: The bulb surface area."""
        return (2 * np.pi * self.bulb_width
                * (self.bulb_width + self.bulb_length))

    @property
    def volume(self):
        r"""units.Quantity: The bulb volume."""
        return np.pi * (self.bulb_width ** 2) * self.bulb_length


class SphereBulb(BulbShapeBase):
    r"""Spherical bulb."""

    _name = 'sphere'

    @property
    def bulb_radius(self):
        r"""units.Quantity: The bulb radius."""
        return self.bulb_width / 2.0

    @property
    def surface_area(self):
        r"""units.Quantity: The bulb surface area."""
        return 4.0 * np.pi * (self.bulb_radius ** 2)

    @property
    def volume(self):
        r"""units.Quantity: The bulb volume."""
        return 4.0 * np.pi * (self.bulb_radius ** 3) / 3.0


class StandardBulb(BulbShapeBase):
    r"""Standard bulb."""
    _arguments = BulbShapeBase._arguments.copy(
        modifications={
            'bulb_width': {'default': units.Quantity(6.1, 'cm')},
        },
    ) + [
        (('--bulb-height', ), {
            'units': 'cm',
            'default': units.Quantity(11.4, 'cm'),
            'help': 'Height of the bulb (in cm)',
        }),
    ]
    # TODO: Area


class PointBulb(BulbShapeBase):
    r"""Single point source."""

    _name = 'point'


class ArtificalLightSourceBase(LightSourceBase):
    r"""Base class for artifical light source."""

    _arguments = [
        arguments.ClassSubparserArgumentDescription('bulb_type'),
        arguments.ClassSubparserArgumentDescription('bulb_shape'),
        (('--light-duration', ), {
            'default': 12.0, 'units': 'hr',
            'help': ('Number of the hours each light should be turned '
                     'on in each illumination cycle.'),
        }),
        (('--light-period', ), {
            'default': 24.0, 'units': 'hr',
            'help': ('Number of the hours between the starts of '
                     'sequential illumination cycles.'),
        }),
    ]
