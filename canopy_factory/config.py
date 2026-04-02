import os
import yggdrasil_rapidjson as rapidjson
from collections import OrderedDict
from configparser import ConfigParser


class PackageConfig(ConfigParser, object):
    r"""Config parser that initializes a default for a package.

    Args:
        package (str): Name of the package the config file is for.
        ext (str, optional): File extension for config files.
        schema (dict, optional): JSON schema that should be used to parse
            the loaded result.
        defaults (dict, optional): Default values that should be
            preloaded.

    """

    def __init__(self, package, ext='.ini', schema=None, defaults=None):
        self.package = package
        self.schema = schema
        self.defaults = defaults
        self.fbase = f'.{package}{ext}'
        self.directories = OrderedDict([
            ('package', os.path.dirname(__file__)),
            # ('user', appdirs.site_config_dir(appname=package)),
            ('user', os.path.expanduser('~')),
            ('local', os.getcwd()),
        ])
        self.files = OrderedDict([
            (k, os.path.join(v, self.fbase))
            for k, v in self.directories.items()
        ])
        super(PackageConfig, self).__init__(
            allow_no_value=True,
            converters={'json': rapidjson.loads},
        )
        self.read()
        if not os.path.isfile(self.files['user']):
            self.write(self.files['user'])

    @property
    def json(self):
        r"""dict: JSON representation of the config options."""
        out = {}
        for section in self.sections():
            out[section] = {
                k: self.get(section, k) for k in self.options(section)
            }
        return out

    def read(self, fnames=None, reset=False):
        r"""Read config options from the default file locations.

        Args:
            fnames (str, list, optional): One or more paths to files that
                should be read instead of the default files. If not
                provided the default files will be read.
            reset (bool, optional): If True, existing values will be
                cleared.

        """
        if reset:
            self.reset()
        if fnames is None:
            fnames = list(self.files.values())
        if self.defaults is not None:
            self.read_dict(self.defaults)
        super(PackageConfig, self).read(fnames)
        if self.schema is not None:
            rapidjson.validate(self.json, self.schema)

    def write(self, fname=None):
        r"""Write the config options to a file.

        Args:
            fname (str, optional): Path to the file that the values should
                be written to. If not provided, the local file will be
                used.

        """
        if fname is None:
            fname = self.files['local']
        with open(fname, 'w') as fd:
            super(PackageConfig, self).write(fd)

    def set(self, section, option, value=None):
        r"""Set an option in a section.

        Args:
            section (str): Section name.
            option (str): Option name.
            value (object, optional): Option value. None indicates an
                empty option value.

        """
        if not (value is None or isinstance(value, str)):
            value = rapidjson.dumps(value)
        return super(PackageConfig, self).set(section, option, value)

    def reset(self):
        r"""Clear the current parameters."""
        for s in self.sections():
            self.remove_section(s)
        self._sections = self._dict()

    def setdefaults(self, **kwargs):
        r"""Add sections/options that do not already exist.

        Args:
            **kwargs: Section/options pairs that should be added if they
                don't exist.

        """
        for section, options in kwargs.items():
            if not self.has_section(section):
                self.add_section(section)
            for k, v in options.items():
                if self.has_option(section, k):
                    continue
                self.set(section, k, v)
