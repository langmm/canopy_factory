import abc
import copy
import argparse
import inspect
from collections import OrderedDict
from canopy_factory import utils
from canopy_factory.utils import (
    cached_property, get_class_registry,
    RegisteredClassBase, NoDefault
)
# TODO: Move output_suffix into argument description


class ArgumentDescriptionABC(abc.ABC):
    r"""Abstract base class providing argument mixin methods.

    Args:
        name (str): Argument name.
        ignored (bool, optional): If True, the argument will be set to
            None during adjustment.
        **kwargs: Additional keyword arguments are passed to the parent
            class.

    """
    _properties_attributes = ['ignored']  # , 'output_suffix']
    _properties_lists = []
    _properties_dicts = []
    _properties_inherit = ['ignored']

    def __init__(self, name, ignored=False, output_suffix=False,
                 **kwargs):
        self._name = name
        self._classes = OrderedDict()
        self.ignored = ignored
        # self._output_suffix = output_suffix
        super(ArgumentDescriptionABC, self).__init__(**kwargs)

    def __repr__(self):
        type_str = str(type(self)).split("'")[1]
        return f'{type_str}({self.name}, dest={self.dest})'

    @property
    def name(self):
        r"""str: Name of the argument."""
        return self._name

    @property
    def dest(self):
        r"""str: Name of variable where the argument will be stored."""
        return self.name

    @property
    def members(self):
        r"""iterable: Set of subarguments."""
        return []

    @property
    def cls(self):
        r"""type: Immediate class that uses this argument."""
        return next(iter(self._classes.values()), None)

    @property
    def top_cls(self):
        r"""type: Top level class that uses this argument."""
        return next(iter(reversed(self._classes.values())), None)

    @property
    def is_output(self):
        r"""bool: True if the argument describes an output."""
        return False

    # @property
    # def output_suffix(self):
    #     r"""ArgumentSuffix: Suffix generator for this argument."""
    #     if self._output_suffix is False:
    #         return None
    #     kwargs = (
    #         self._output_suffix if isinstance(self._output_suffix, dict)
    #         else {}
    #     )
    #     assert self.dest is not None
    #     return ArgumentSuffix(self.dest, arg=self, **kwargs)

    def flatten(self, no_root=False):
        r"""Iterate over all basic arguments (non-containers).

        Args:
            no_root (bool, optional): If True, don't include this
                argument.

        Yields:
            ArgumentDescription: Arguments that are not a container.

        """
        if isinstance(self, ArgumentDescription) and not no_root:
            yield self
        for v in self.members:
            for x in v.flatten():
                yield x

    def getnested(self, k, default=NoDefault):
        r"""Retrieve an argument that may be nested inside another
        argument set.

        Args:
            k (str): Key name to find an argument for.
            default (object, optional): Value to return if an argument
                cannot be located.

        Returns:
            ArgumentDescription: Argument description instance.

        """
        if self.dest is not None and k == self.dest:
            return self
        for v in self.members:
            out = v.getnested(k, None)
            if out is not None:
                return out
        if default is not NoDefault:
            return default
        raise KeyError(k)

    def argument_names(self, include='dest', no_root=False):
        r"""Get all fo the associated argument names.

        Args:
            include (str, optional): Specifies what to include in the
                returned list. Supported options include:
                  'dest': Full argument name with prefix/suffix.
                  'name': Argument name without prefix/suffix.
                  'both': Tuple of result from 'dest' & 'name'.
                  'flag': Use the first CLI flag.
            no_root (bool, optional): If True, don't include this
                argument.

        Returns:
            list: Argument names.

        """
        out = []
        if self.dest and not no_root:
            if include == 'dest':
                out.append(self.dest)
            elif include == 'name':
                out.append(self.name)
            elif include == 'both':
                out.append((self.name, self.dest))
            elif include == 'flag':
                out.append(self.keys[0])
            else:
                raise ValueError(include)
        for v in self.members:
            out += [
                x for x in v.argument_names(include=include)
                if x not in out
            ]
        return out

    def copy(self, **kwargs):
        r"""Create a copy of this argument set with modifications.

        Args:
            **kwargs: Keyword arguments are passed to the modify method
                for the returned copy.

        Returns:
            ArgumentDescriptionSet: New copy of the argument set with
                modifications.

        """
        out = copy.deepcopy(self)
        out.modify(**kwargs)
        return out

    def any_arguments_set(self, args):
        r"""Check if any arguments are set.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            bool: True if arguments are set.

        """
        if ((self.dest and (not self.ignored)
             and getattr(args, self.dest, None) is not None)):
            return True
        for v in self.members:
            if v.any_arguments_set(args):
                return True
        return False

    def _modify(self, name, value, src=None):
        r"""Modify an argument property.

        Args:
            name (str): Property name. If the name starts with 'append_',
                the value will be added to the existing values.
            value (object): New property value.
            src (dict, optional): Dictionary for properties in
                self._properties_dicts should be stored.

        Returns:
            bool: True if the property should also be passed to child
                members.

        """
        if name == 'strip_classes':
            if not value:
                return False
            assert value in ['all', True]
            if value == 'all':
                self._classes.clear()
                return True
            else:
                for k in list(self._classes.keys()):
                    self.remove_class(k)
                return False
        elif name == 'add_class':
            self.add_class(value)
            return False
        elif name == 'remove_class':
            self.remove_class(value)
            return False
        if name.startswith('append_'):
            name = name.split('append_', maxsplit=1)[-1]
            assert name in self._properties_lists
            if name in self._properties_lists:
                if not isinstance(value, list):
                    value = [value]
                if name in self._properties_attributes:
                    value = list(getattr(self, name)) + value
                else:
                    if src is None:
                        return True
                    value = list(src.get(name, [])) + value
            elif name in self._properties_dicts:
                assert isinstance(value, dict)
                if name in self._properties_attributes:
                    value = dict(getattr(self, name), **value)
                else:
                    if src is None:
                        return True
                    value = dict(src.get(name, {}), **value)
        if name in self._properties_lists and not isinstance(value, list):
            value = [value]
        if name == 'keys':
            value = tuple(value)
        if name in self._properties_attributes:
            setattr(self, name, value)
        else:
            if src is None:
                return True
            src[name] = value
        return (name in self._properties_inherit)

    def modify(self, modifications=None, include=None, exclude=None,
               ignore=None, **kwargs):
        r"""Modify this argument.

        Args:
            modifications (dict, optional): Mapping between argument
                names and the modifications that should be made to the
                arguments.
            include (list, optional): Subset of arguments to retain.
                Other arguments will be removed.
            exclude (list, optional): Subset of arguments to remove.
            ignore (list, optional): Subset of arguments to ignore.
            **kwargs: Additional keyword arguments are used to update
                member argument properties.

        """
        name = self.dest
        if name is not None:
            if modifications and name in modifications:
                kwargs.update(**modifications[name])
            if include is not None:
                assert name in include
            if exclude is not None:
                assert name not in exclude
            if ignore is not None and name in ignore:
                kwargs['ignored'] = True
        kwargs_members = {}
        for k, v in kwargs.items():
            if self._modify(k, v):
                kwargs_members[k] = v
        for v in self.members:
            v.modify(
                modifications=modifications, include=include,
                exclude=exclude, ignore=ignore, **kwargs_members
            )

    def add_to_parser(self, parser, only_subparsers=False, **kwargs):
        r"""Add this argument to a parser.

        Args:
            parser (InstrumentedParser): Parser that the argument should
                be added to.
            only_subparsers (bool, optional): If True, only add
                add subparsers.
            **kwargs: Additional keyword arguments are passed to
                member's add_to_parser methods.

        """
        if self.ignored:
            return
        for v in self.members:
            v.add_to_parser(parser, only_subparsers=True, **kwargs)
        if only_subparsers:
            return
        for v in self.members:
            v.add_to_parser(parser, only_subparsers=False, **kwargs)

    @classmethod
    def kwargs2args(cls, kwargs, include=None):
        r"""Convert from a set of keyword arguments to an argument
        namespace.

        Args:
            kwargs (dict): Keyword arguments.
            include (list, optional): Names of arguments to include.

        Returns:
            argparse.Namespace: Argument namespace.

        """
        out = argparse.Namespace()
        if include is not None:
            for k in include:
                setattr(out, k, kwargs.get(k, None))
        else:
            for k, v in kwargs.items():
                setattr(out, k, v)
        return out

    @classmethod
    def args2kwargs(cls, args, include=None):
        r"""Convert arguments from an argument namespace to keyword
        arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            include (list, optional): Names of arguments to include.

        Returns:
            dict: Keyword arguments.

        """
        if include is not None:
            return {k: getattr(args, k, None) for k in include}
        return vars(args)

    def extract_args(self, args, out=None):
        r"""Extract arguments that are set into a dictionary.

        Args:
            args (argparse.Namespace): Parsed arguments.
            out (dict, optional): Existing dictionary that arguments
                should be added to.

        Returns:
            dict: Updated argument dictionary.

        """
        if out is None:
            out = {}
        name = self.dest
        if ((name and name not in out
             and getattr(args, name, None) is not None)):
            out[name] = getattr(args, name)
        for v in self.members:
            v.extract_args(args, out=out)
        return out

    @abc.abstractmethod
    def from_args(self, args):
        r"""Construct the argument from an argument namespace.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            object: Argument value.

        """
        raise NotImplementedError

    def from_kwargs(self, kws, **kwargs):
        r"""Construct the argument from a dictionary of keyword arguments.

        Args:
            kws (dict): Keyword arguments.
            **kwargs: Additional keyword arguments are passed to
                from_args.

        Returns:
            object: Argument value.

        """
        args = self.kwargs2args(kws)
        return self.from_args(args, **kwargs)

    def finalize(self, x, args=None):
        r"""Finalize an argument.

        Args:
            x (object): Argument value to finalize.
            args (argparse.Namespace, optional): Parsed arguments.

        Returns:
            object: The finalized instance.

        """
        return x

    def parse(self, x, args=None, **kwargs):
        r"""Parse an argument.

        Args:
            x (object): Argument value to parse.
            args (argparse.Namespace, optional): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to the
                from_args method.

        Returns:
            object: The parsed instance.

        """
        if args is None:
            return self.finalize(x)
        args = copy.deepcopy(args)
        self.reset_args(args, value=x)
        return self.from_args(args, **kwargs)

    def reset_args(self, args, value=NoDefault):
        r"""Reset the arguments for a variable.

        Args:
            args (argparse.Namespace): Parsed arguments.
            value (object, optional): Value that the parsed variable
                should be reset to.

        """
        name = self.dest
        if name is not None:
            if value is NoDefault:
                value = getattr(args, name, None)
                if isinstance(value, RegisteredArgumentClassBase):
                    value = value.getarg(name, None)
            setattr(args, name, value)
        for v in self.members:
            v.reset_args(args)

    def adjust_args(self, args, skip=None, skip_root=False,
                    skip_members=False, members=None, overwrite=False,
                    **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            skip (list, optional): Arguments to skip.
            skip_root (bool, optional): If True, don't adjust the base
                argument.
            skip_members (bool, optional): If True, don't adjust the
                member arguments.
            members (list, optional): Subset of members to adjust
                arguments for.
            overwrite (bool, optional): If True, overwrite the existing
                argument value.
            **kwargs: Additional keyword arguments are passed to
                from_args.

        """
        if skip is None:
            skip = []
        if overwrite:
            self.reset_args(args)
        if self.dest is not None and self.dest in skip:
            return
        if not skip_members:
            if members is None:
                members = self.members
            for v in members:
                v.adjust_args(args, skip=skip)
        if skip_root or self.dest is None:
            return
        if isinstance(getattr(args, self.dest, None),
                      RegisteredArgumentClassBase):
            return
        setattr(args, self.dest, self.from_args(args, **kwargs))

    def adjust_kwargs(self, kws, **kwargs):
        r"""Adjust an arguments dictionary including setting defaults
        that depend on other provided arguments.

        Args:
            kws (dict): Arguments.
            **kwargs: Additional keyword arguments are passed to
                adjust_args.

        """
        args = self.kwargs2args(kws)
        self.adjust_args(args, **kwargs)
        for k, v in self.args2kwargs(args).items():
            kws[k] = v

    def remove_class(self, cls):
        r"""Remove a class that uses the argument(s).

        Args:
            cls (type, str): Class that uses the argument(s) or the
                registry key for the class type that should be removed.

        """
        if isinstance(cls, type):
            cls = cls._registry_key
        if cls is None:
            return
        assert cls in self._classes
        self._classes.pop(cls)
        for v in self.members:
            v.remove_class(cls)

    def add_class(self, cls, overwrite=False):
        r"""Add a class that uses the argument(s).

        Args:
            cls (type): Class that uses the argument(s).
            overwrite (bool, optional): If True, overwrite any existing
                class for the specified key.

        """
        if not overwrite:
            assert cls._registry_key not in self._classes
        self._classes[cls._registry_key] = cls
        for v in self.members:
            v.add_class(cls, overwrite=overwrite)


class ArgumentDescription(ArgumentDescriptionABC):
    r"""Base class for describing a CLI argument.

    Args:
        keys (tuple): Set of CLI keys.
        properties (dict): Keyword arguments to parser add_argument
            describing the argument and how it should be interpreted.
        prefix (str, optional): Prefix that should be added to the
            argument name.
        suffix (str, optional): Suffix that should be added to the
            argument name.
        description (str, optional): Description string that should be
            used to format the help message.
        universal (bool, optional): If True, the argument should only
            be added to a parser if it dosn't already have an argument of
            that name.
        **kwargs: Additional keyword arguments are passed to the parent
            class.

    """

    _properties_attributes = (
        ArgumentDescriptionABC._properties_attributes + [
            'keys', 'class_converter', 'upstream', 'universal',
            'subparser_specific_dest',
            'prefix', 'suffix', 'description',
            'units', 'units_arg',
        ]
    )
    _properties_lists = (
        ArgumentDescriptionABC._properties_lists + [
            'keys', 'choices',
        ]
    )

    def __init__(self, keys, properties, prefix='', suffix='',
                 description='', universal=False, **kwargs):
        name = properties.get(
            'dest', keys[0].lstrip('--').replace('-', '_'))
        self._keys = keys
        self.class_converter = properties.pop('class_converter', None)
        self.units = properties.pop('units', None)
        self.units_arg = properties.pop('units_arg', None)
        self._class_converter_method = None
        self.upstream = properties.pop('upstream', [])
        self.subparser_specific_dest = properties.pop(
            'subparser_specific_dest', False)
        self.properties = properties
        self.prefix = prefix
        self.suffix = suffix
        self.description = description
        self.universal = universal
        if self.subparser_specific_dest:
            assert 'dest' not in self.properties
        if self.units:
            assert 'type' not in self.properties
            self.properties.setdefault(
                'type', utils.QuantityArgument(
                    self.units, self.properties.pop('choices', None)))
        elif isinstance(self.properties.get('type', None),
                        utils.QuantityArgument):
            self.units = self.properties['type'].default_units
        super(ArgumentDescription, self).__init__(name, **kwargs)
        if self.units and 'default' in self.properties:
            self.properties['default'] = self.finalize(
                self.properties['default'])

    @property
    def subparser(self):
        r"""str: Name of subparser used in destination."""
        if isinstance(self.subparser_specific_dest, str):
            return self.subparser_specific_dest
        elif self.subparser_specific_dest is True and self.top_cls:
            return self.top_cls._name
        return None

    @property
    def prefix_arg(self):
        r"""str: Prefix to add to argument keys."""
        return self.prefix.replace('_', '-')

    @property
    def suffix_arg(self):
        r"""str: Suffix to add to argument keys."""
        return self.suffix.replace('_', '-')

    @property
    def name(self):
        r"""str: Name of the argument."""
        out = self._name
        subparser = self.subparser
        if isinstance(subparser, str):
            out += f'_{subparser}'
        return out

    @property
    def dest(self):
        r"""str: Name of variable where the argument will be stored."""
        if 'dest' in self.properties:
            return self.properties['dest']
        return self.prefix + self.name + self.suffix

    @property
    def keys(self):
        r"""tuple: Set of keys for the argument."""
        if not (self.subparser or self.prefix or self.suffix):
            return self._keys
        prefix_arg = self.prefix.replace('_', '-')
        suffix_arg = self.suffix.replace('_', '-')
        if self.subparser:
            suffix_arg += f'_{self.subparser}'
        return tuple([
            f'--{prefix_arg}' + k.split('--', 1)[-1] + suffix_arg
            for k in self._keys if k.startswith('--')
        ])

    @cached_property
    def default(self):
        r"""object: Argument default."""
        out = None
        if self.properties.get('action', None) == 'store_true':
            return False
        elif self.properties.get('action', None) == 'store_false':
            return True
        elif 'default' in self.properties:
            out = self.properties['default']
        return out

    @classmethod
    def get_class_converter_method(cls, src, name):
        r"""Locate a class converter method on the source class.

        Args:
            src (type): Source class.
            name (str): Conversion name.

        Returns:
            callable: Converter method.

        """
        return getattr(src, f'_converter_{name}', None)

    @property
    def class_converter_method(self):
        r"""str: Name of the class method that should be used to
        finalize the argument."""
        if self._class_converter_method is not None:
            return self._class_converter_method
        if not self.class_converter:
            return None
        return self.get_class_converter_method(self.top_cls,
                                               self.class_converter)

    def _modify(self, name, value):
        if name == 'subparser':
            if self.subparser_specific_dest is True:
                self.subparser_specific_dest = value
            return True
        if ((name == 'class_converter'
             and value != self._class_converter_method)):
            self._class_converter_method = None
        return super(ArgumentDescription, self)._modify(
            name, value, self.properties)

    def add_to_parser(self, parser, only_subparsers=False, **kwargs):
        r"""Add this argument to a parser.

        Args:
            parser (InstrumentedParser): Parser that the argument should
                be added to.
            only_subparsers (bool, optional): If True, only add
                add subparsers.
            **kwargs: Additional keyword arguments are passed to
                member's add_to_parser methods.

        """
        super(ArgumentDescription, self).add_to_parser(
            parser, only_subparsers=only_subparsers, **kwargs)
        if only_subparsers or self.ignored:
            return
        kws = dict(self.properties)
        kws.setdefault('dest', self.dest)
        if self.prefix or self.suffix or self.description:
            prefix_dst = self.prefix
            suffix_dst = self.suffix
            prefix_arg = prefix_dst.replace('_', '-')
            suffix_arg = suffix_dst.replace('_', '-')
            if 'help' in kws:
                kws['help'] = kws['help'].format(
                    description=self.description,
                    prefix_arg=prefix_arg, prefix_dst=prefix_dst,
                    suffix_arg=suffix_arg, suffix_dst=suffix_dst,
                )
                if self.units and self.units not in kws['help']:
                    kws['help'] = (
                        kws['help'].rstrip('.')
                        + f' (in {self.units}).'
                    )
            if 'dest' in kws:
                assert '{prefix_dst}' not in kws['dest']
                assert '{suffix_dst}' not in kws['dest']
                # kws['dest'] = kws['dest'].format(
                #     prefix_dst=prefix_dst,
                #     suffix_dst=suffix_dst,
                # )
        if self.universal and parser.has_argument(self.dest):
            return
        parser.add_argument(*self.keys, **kws)

    def from_args(self, args):
        r"""Construct the argument from an argument namespace.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            object: Argument value.

        """
        if self.ignored:
            return None
        out = getattr(args, self.dest, self.default)
        return self.finalize(out, args=args)

    def finalize(self, x, args=None):
        r"""Finalize an argument.

        Args:
            x (object): Argument value to finalize.
            args (argparse.Namespace, optional): Parsed arguments.

        Returns:
            object: The finalized instance.

        """
        if 'type' in self.properties and x is not None:
            x = self.properties['type'](x)
        if args is not None:
            if self.units_arg and getattr(args, self.units_arg, None):
                x = utils.parse_quantity(x, getattr(args, self.units_arg))
            if self.class_converter is not None:
                method = self.class_converter_method
                if method is None:
                    raise ValueError(
                        f'{self.dest}: Could not located a conversion '
                        f'method \"{self.class_converter}\" (top class = '
                        f'{self.top_cls})')
                x = method(args, x)
        return super(ArgumentDescription, self).finalize(x, args=args)

    def add_class(self, cls, overwrite=False):
        r"""Add a class that uses the argument(s).

        Args:
            cls (type): Class that uses the argument(s).
            overwrite (bool, optional): If True, overwrite any existing
                class for the specified key.

        """
        super(ArgumentDescription, self).add_class(
            cls, overwrite=overwrite)
        if self.class_converter:
            new_method = self.get_class_converter_method(
                cls, self.class_converter)
            if new_method:
                self._class_converter_method = new_method


class SubparserArgumentDescription(ArgumentDescription):
    r"""Subparser argument set.

    Args:
        name (str): Name of the subparser.
        properties (dict): Keyword arguments for creating the subparser
            group.
        subparser_properties (dict, optional): Mapping between subparser
            names and properties specific to the creation of that
            subparser.
        subparser_arguments (dict, optional): Mapping between subparser
            names and sets of arguments for that subparser.
        using_flag (bool, optional): If True, a standard option argument
            should be used as a switch for the different subparsers.

    """

    _properties_attributes = (
        ArgumentDescription._properties_attributes + [
            'using_flag',
        ]
    )

    def __init__(self, name, properties, subparser_properties=None,
                 subparser_arguments=None, using_flag=False):
        keys = (name.replace('_', '-'), )
        if subparser_properties is None:
            subparser_properties = {}
        if subparser_arguments is None:
            subparser_arguments = {}
        self.using_flag = using_flag
        self.subparser_properties = subparser_properties
        self.subparser_arguments = {
            k: ArgumentDescriptionSet(v) if isinstance(v, list) else v
            for k, v in subparser_arguments.items()
        }
        for k, v in self.subparser_arguments.items():
            v.modify(subparser=k)
        super(SubparserArgumentDescription, self).__init__(
            keys, properties)

    @property
    def members(self):
        r"""iterable: Set of subarguments."""
        return self.subparser_arguments.values()

    def add_to_parser(self, parser, only_subparsers=False, **kwargs):
        r"""Add this argument to a parser.

        Args:
            parser (InstrumentedParser): Parser that the argument should
                be added to.
            only_subparsers (bool, optional): If True, only add
                add subparsers.
            **kwargs: Additional keyword arguments are passed to
                member's add_to_parser methods.

        """
        if self.ignored:
            return
        assert 'choices' in self.properties
        if self.using_flag:
            super(SubparserArgumentDescription, self).add_to_parser(
                parser, only_subparsers=False, **kwargs
            )
            return
        subparsers = {}
        for k in self.properties['choices']:
            add_missing = self.subparser_properties.get(k, {})
            add_missing_group = {
                k: v for k, v in self.properties.items()
                if k not in ['choices']
            }
            subparsers[k] = parser.get_subparser(
                self.dest, k,
                add_missing_group=add_missing_group,
                add_missing=add_missing,
            )
        for k, v in self.subparser_arguments.items():
            v.add_to_parser(subparsers[k], only_subparsers=True,
                            **kwargs)
        if only_subparsers:
            return
        for k, v in self.subparser_arguments.items():
            v.add_to_parser(subparsers[k], only_subparsers=False,
                            **kwargs)

    def adjust_args(self, args, **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            **kwargs: Additional keyword arguments are passed to the
                parent class's method.

        """
        members = None
        if not (self.ignored or self.using_flag):
            subparser = super(SubparserArgumentDescription,
                              self).from_args(args)
            if isinstance(subparser, RegisteredArgumentClassBase):
                subparser = subparser._name
            members = [self.subparser_arguments[subparser]]
        super(SubparserArgumentDescription, self).adjust_args(
            args, members=members, **kwargs)

    def from_args(self, args, **kwargs):
        r"""Construct the argument from an argument namespace.

        Args:
            args (argparse.Namespace): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to subparser
                from_args method.

        Returns:
            object: Argument value.

        """
        subparser = super(SubparserArgumentDescription,
                          self).from_args(args)
        if ((subparser not in self.subparser_arguments
             or (not self.subparser_arguments[subparser].cls)
             or self.subparser_arguments[subparser].dont_create)):
            return subparser
        return self.subparser_arguments[subparser].from_args(args, **kwargs)


class ClassSubparserArgumentDescription(SubparserArgumentDescription):
    r"""Class for adding arguments associated with a set of registered
    classes as subparsers.

    Args:
        name (str): Name of the registered class set to create subparsers
            for.
        properties (dict, optional): Keyword arguments for creating the
            subparser group.
        subparser_properties (dict, optional): Mapping between subparser
            names and properties specific to the creation of that
            subparser.
        subparser_arguments (dict, optional): Mapping between subparser
            names and sets of arguments for that subparser.
        modifications (dict, optional): Mapping between argument names
            and the modifications that should be made to the
            arguments.
        include (list, optional): Subset of arguments to retain.
            Other arguments will be removed.
        exclude (list, optional): Subset of arguments to remove.
        dont_create (bool, optional): If True, the argument will not be
            created when it is parsed via from_args.

    """

    def __init__(self, name, properties=None, subparser_properties=None,
                 subparser_arguments=None, modifications=None,
                 include=None, exclude=None, dont_create=False):
        if isinstance(name, type):
            base_class = name
            name = base_class._registry_key
        else:
            base_class = get_class_registry().getbase(name)
        assert issubclass(base_class, RegisteredArgumentClassBase)
        if properties is None:
            properties = {}
        if subparser_properties is None:
            subparser_properties = {}
        if subparser_arguments is None:
            subparser_arguments = {}
        properties.setdefault(
            'choices', list(get_class_registry().keys(name)))
        if base_class._default:
            properties.setdefault('default', base_class._default)
        if base_class._help:
            properties.setdefault('help', base_class._help)
        for k in properties['choices']:
            v = get_class_registry().get(name, k)
            subparser_arguments[k] = v._arguments.copy()
            subparser_arguments[k].dont_create = dont_create
            # Should already be present
            # subparser_arguments[k].add_class(v)
            if v._help:
                subparser_properties.setdefault(k, {})
                subparser_properties[k].setdefault('help', v._help)
        super(ClassSubparserArgumentDescription, self).__init__(
            name, properties, subparser_properties=subparser_properties,
            subparser_arguments=subparser_arguments)
        if include is not None:
            include = [self.dest] + include
        self.modify(modifications=modifications,
                    include=include, exclude=exclude)


class ArgumentDescriptionSet(ArgumentDescriptionABC, utils.SimpleWrapper):
    r"""A set of CLI arguments.

    Args:
        arguments (list, optional): Argument descriptions.
        name (str, optional): Name to give the argument set.
        dont_create (bool, optional): If True, don't create an instance
            of the argument class containing this argument set during
            calls to from_args.
        prefix (str, optional): Prefix that should be added to the
            argument name.
        suffix (str, optional): Suffix that should be added to the
            argument name.
        description (str, optional): Description string that should be
            used to format the help message.
        ignore (list, optional): Subset of arguments to ignore.
        **kwargs: Additional keyword arguments will be passed to the
            parent class constructor in from_args.

    """

    _properties_attributes = (
        ArgumentDescriptionABC._properties_attributes + [
            'cls_kwargs', 'dont_create',
        ]
    )
    _properties_dicts = (
        ArgumentDescriptionABC._properties_dicts + [
            'cls_kwargs',
        ]
    )

    def __init__(self, arguments=None, name=None, dont_create=False,
                 **kwargs):
        if arguments is None:
            arguments = []
        arg_kwargs = {
            k: kwargs.pop(k) for k in [
                'prefix', 'suffix', 'description', 'ignore'
            ] if k in kwargs
        }
        self.cls_kwargs = kwargs
        self.dont_create = dont_create
        super(ArgumentDescriptionSet, self).__init__(name, ordered=True)
        if not isinstance(arguments, list):
            arguments = [arguments]
        for x in arguments:
            self.append(x, **arg_kwargs)

    @property
    def members(self):
        r"""iterable: Set of subarguments."""
        return self.values()

    def modify(self, modifications=None, include=None, exclude=None,
               **kwargs):
        r"""Modify this argument.

        Args:
            modifications (dict, optional): Mapping between argument
                names and the modifications that should be made to the
                arguments.
            include (list, optional): Subset of arguments to retain.
                Other arguments will be removed.
            exclude (list, optional): Subset of arguments to remove.
            **kwargs: Additional keyword arguments are used to update
                individual argument properties.

        """
        remove = set()
        if isinstance(exclude, list):
            dests = self.dests
            remove |= set([dests[k].name for k in exclude if k in dests])
        if isinstance(include, list):
            remove |= set([v.name for v in self.values()
                           if v.dest not in include])
        for k in remove:
            del self[k]
        super(ArgumentDescriptionSet, self).modify(
            modifications=modifications, include=include,
            exclude=exclude, **kwargs)
        if ((kwargs.get('strip_classes')
             or any(k in kwargs for k in ['subparser', 'add_class',
                                          'remove_class']))):
            self.reset_keys()

    def from_args(self, args, cls=None, **kwargs):
        r"""Construct the argument from an argument namespace.

        Args:
            args (argparse.Namespace): Parsed arguments.
            cls (type, optional): Class that should be constructed from
                the arguments. Defaults to the 'cls' attribute.
            **kwargs: Additional keyword arguments are passed to cls
                if one is provided.

        Returns:
            object: Argument value or argument namespace
                (if the argument set does not have a cls set).

        """
        if cls is None and not self.dont_create:
            cls = self.cls
        if cls is None:
            # return {v.dest: v.from_args(args) for v in self.values()}
            raise RuntimeError(f'No class for argument set '
                               f'\"{self.dest}\"')
        if self.cls_kwargs:
            kwargs = dict(self.cls_kwargs, **kwargs)
        return cls.from_args(args, **kwargs)

    def __add__(self, solf):
        out = ArgumentDescriptionSet()
        out.append(self)
        out.append(solf)
        return out

    def __radd__(self, solf):
        out = ArgumentDescriptionSet()
        out.append(solf)
        out.append(self)
        return out

    def append(self, x, **kwargs):
        r"""Add an argument to the argument set.

        Args:
            x (ArgumentDescription, tuple): Argument to add.
            **kwargs: Additional keyword arguments will be used to
                modify a copy of x prior to adding it.

        """
        if isinstance(x, ArgumentDescriptionSet) and x.name is None:
            kwargs.setdefault('strip_classes', True)
            for v in x.values():
                self.append(v, **kwargs)
            return
        elif isinstance(x, ArgumentDescriptionABC):
            x = x.copy(**kwargs)
        elif isinstance(x, tuple):
            x = ArgumentDescription(*x)
            if kwargs:
                x.modify(**kwargs)
        elif isinstance(x, list):
            for xx in x:
                self.append(xx, **kwargs)
            return
        self[x.name] = x

    def __setitem__(self, k, v, dont_add_class=False):
        if k in self:
            if self[k] == v:
                return
            raise KeyError(f"Cannot replace argument \"{k}\"")
        if not isinstance(v, ArgumentDescriptionABC):
            raise TypeError(f'ArgumentDescription required, not '
                            f'\"{type(v)}\"')
        if not dont_add_class:
            for cls in self._classes.values():
                v.add_class(cls)
        super(ArgumentDescriptionSet, self).__setitem__(k, v)

    def remove_class(self, cls):
        r"""Remove a class that uses the argument(s).

        Args:
            cls (type, str): Class that uses the argument(s) or the
                registry key for the class type that should be removed.

        """
        super(ArgumentDescriptionSet, self).remove_class(cls)
        self.reset_keys()

    def add_class(self, cls, overwrite=False):
        r"""Add a class that uses the argument(s).

        Args:
            cls (type): Class that uses the argument(s).
            overwrite (bool, optional): If True, overwrite any existing
                class for the specified key.

        """
        super(ArgumentDescriptionSet, self).add_class(
            cls, overwrite=overwrite)
        self.reset_keys()

    @property
    def dests(self):
        r"""dict: Mapping between destination variable name and
        argument."""
        return {v.dest: v for v in self.values()}

    # def getnested(self, k, default=NoDefault):
    #     r"""Retrieve an argument that may be nested inside another
    #     argument set.

    #     Args:
    #         k (str): Key name to find an argument for.
    #         default (object, optional): Value to return if an argument
    #             cannot be located.

    #     Returns:
    #         ArgumentDescription: Argument description instance.

    #     """
    #     if k in self:
    #         return self[k]
    #     return super(ArgumentDescriptionSet, self).getnested(
    #         k, default=default)

    def getkey(self, k, default=NoDefault):
        r"""Retrieve an argument based on a key value.

        Args:
            k (str): Key name to find an argument for.
            default (object, optional): Value to return if an argument
                cannot be located.

        Returns:
            ArgumentDescription: Argument description instance.

        """
        for v in self.values():
            if k in v.keys:
                return v
        if default is not NoDefault:
            return default
        raise KeyError(k)

    def getdest(self, k, default=NoDefault):
        r"""Retrieve an argument based on the destination variable.

        Args:
            k (str): Destination name to find an argument for.
            default (object, optional): Value to return if an argument
                cannot be located.

        Returns:
            ArgumentDescription: Argument description instance.

        """
        for v in self.values():
            if v.dest == k:
                return v
        if default is not NoDefault:
            return default
        raise KeyError(k)

    def reset_keys(self):
        r"""Reinitialize the member arguments to allow the regeneration
        of argument keys (destination names)."""
        values = list(self.values())
        self.clear()
        for v in values:
            self.__setitem__(v.name, v, dont_add_class=True)

    # @property
    # def output_suffix(self):
    #     r"""ArgumentSuffix: Suffix generator for this argument."""
    #     if self._output_suffix is False:
    #         return None
    #     assert self.dest is None
    #     kwargs = (
    #         self._output_suffix if isinstance(self._output_suffix, dict)
    #         else {}
    #     )
    #     out = ArgumentSetSuffix([], **kwargs)
    #     for v in self.values():
    #         vsuffix = v.output_suffix
    #         if vsuffix:
    #             out.arguments[vsuffix.name] = vsuffix
    #     return out


class CompositeArgumentDescription(ArgumentDescriptionSet):
    r"""Composite argument.

    Args:
        name (str): Name of the composite argument.
        composite_type (str, optional): Name of the registered
            composite class that should be created for this argument.
            Defaults to name if not provided.
        **kwargs: Additional keyword arguments will be passed to the
            ArgumentDescriptionSet constructor.

    """

    def __init__(self, name, composite_type=None, prefix='', suffix='',
                 **kwargs):
        if composite_type is None:
            composite_type = name
        name = prefix + name + suffix
        cls_base = get_class_registry().get('argument', composite_type)
        kwargs_cls = {
            k: kwargs.pop(k) for k in [
                'description', 'ignore', 'registry_name'
            ] if k in kwargs
        }
        cls = cls_base.class_factory(name, **kwargs_cls)
        super(CompositeArgumentDescription, self).__init__(
            cls._arguments, name=name, **kwargs)
        self.add_class(cls, overwrite=True)

    @property
    def is_output(self):
        r"""bool: True if the argument describes an output."""
        return (self.cls._name == 'output')

    def argument_names(self, include='dest'):
        r"""Get all fo the associated argument names.

        Args:
            include (str, optional): Specifies what to include in the
                returned list. Supported options include:
                  'dest': Full argument name with prefix/suffix.
                  'name': Argument name without prefix/suffix.
                  'both': Tuple of result from 'dest' & 'name'.
                  'flag': Use the first CLI flag.

        Returns:
            list: Argument names.

        """
        if include == 'prefixed':
            out = []
            for v in self.members:
                if v.name in self.cls._arguments_prefixed:
                    out += [
                        x for x in v.argument_names()
                        if x not in out
                    ]
            return out
        elif include == 'universal':
            out = []
            for v in self.members:
                if v.name in self.cls._arguments_universal:
                    out += [
                        x for x in v.argument_names()
                        if x not in out
                    ]
            return out
        return super(CompositeArgumentDescription, self).argument_names(
            include=include)

    @property
    def _arguments_prefixed(self):
        r"""ArgumentDescriptionSet: Prefixed arguments."""
        return self.cls._arguments_prefixed

    @property
    def _arguments_universal(self):
        r"""ArgumentDescriptionSet: Unprefixed arguments."""
        return self.cls._arguments_universal


class RegisteredArgumentClassBase(RegisteredClassBase):
    r"""Base class for a class with CLI arguments.

    Args:
        args (argparse.Namespace): Parsed arguments.
        args_overwrite (dict, optional): Arguments to overwrite.

    Class Attributes:
        _arguments (ArgumentDescriptionSet): Descriptions of arguments
            used by this class.

    """

    _name = None
    _help = None
    _default = None
    _arguments = ArgumentDescriptionSet()
    _args_type = 'namespace'

    def __init__(self, args=None, args_overwrite=None):
        # self.adjust_args(args, skip_root=True)
        if self._args_type is None:
            pass
        elif self._args_type == 'namespace':
            self.args = args
        elif self._args_type == 'dict':
            self.args = {}
        elif self._args_type == 'attributes':
            pass
        else:
            raise ValueError(self._args_type)
        super(RegisteredArgumentClassBase, self).__init__()
        self.reset(args, args_overwrite=args_overwrite)

    @staticmethod
    def _on_registration(cls):
        RegisteredClassBase._on_registration(cls)
        if inspect.getmro(cls)[1] == RegisteredClassBase:
            return
        cls._build_arguments(cls)
        if hasattr(cls, '_output_suffix'):
            cls._output_suffix.set_arguments(cls._arguments)

    @staticmethod
    def _build_arguments(cls):
        if isinstance(cls._arguments, list):
            cls._arguments = ArgumentDescriptionSet(cls._arguments)
        cls._arguments = cls._arguments.copy(strip_classes=True)
        if cls._name is not None:
            cls._arguments.add_class(cls)
            assert cls._arguments.cls == cls

    @classmethod
    def adjust_args(cls, args, **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to the
                argument description's adjust_args method.

        """
        cls._arguments.adjust_args(args, **kwargs)

    def reset(self, args, args_overwrite=None):
        r"""Reinitialize the arguments used by this instance.

        Args:
            args (argparse.Namespace): Parsed arguments.
            args_overwrite (dict, optional): Arguments to overwrite.

        """
        if args_overwrite is None:
            args_overwrite = {}
        self.clear_cached_properties()
        if self._args_type is None:
            return
        elif self._args_type == 'namespace':
            self.args = args
        else:
            kwargs = {
                k: (
                    None if self.ignored(k)
                    else args_overwrite.get(kbook,
                                            getattr(args, kbook, None))
                )
                for k, kbook in self.argument_names(include='both')
            }
            if self._args_type == 'dict':
                self.args.clear()
                self.args.update(**kwargs)
            elif self._args_type == 'attributes':
                for k, v in kwargs.items():
                    setattr(self, k, v)
            else:
                raise ValueError(self._args_type)

    def getarg(self, name, default=NoDefault):
        r"""Get an argument value.

        Args:
            name (str): Argument name.
            default (object, optional): Value to return if argument
                does not exist.

        Returns:
            object: Argument value.

        Raises:
            KeyError: If the argument does not exist and default is not
                provided.

        """
        if self._args_type == 'namespace':
            out = getattr(self.args, name, default)
        elif self._args_type == 'dict':
            out = self.args.get(name, default)
        elif self._args_type == 'attributes':
            out = getattr(self, name, default)
        if out is NoDefault:
            raise KeyError(name)
        return out

    @classmethod
    def from_args(cls, args, overwrite=False, **kwargs):
        r"""Construct the class from a argument namespace.

        Args:
            args (argparse.Namespace): Parsed arguments.
            overwrite (bool, optional): If True, overwrite the existing
                instance.
            **kwargs: Additional keyword arguments are passed to the
                class constructor.

        Returns:
            cls: Instance of class.

        """
        if overwrite:
            cls._arguments.reset_args(args)
        name = getattr(args, cls._registry_key, cls._name)
        if isinstance(name, RegisteredArgumentClassBase):
            return name
        assert isinstance(name, str)
        if name != cls._name:
            return get_class_registry().get(
                cls._registry_key, name).from_args(args, **kwargs)
        return cls(args, **kwargs)

    @classmethod
    def from_kwargs(cls, kws, **kwargs):
        r"""Construct the class from a dictionary of keyword arguments.

        Args:
            kws (dict): Keyword arguments.
            **kwargs: Additional keyword arguments are passed to the
                class constructor.

        Returns:
            cls: Instance of class.

        """
        return cls.from_args(cls._arguments.kwargs2args(kws), **kwargs)

    @classmethod
    def argument_names(cls, **kwargs):
        r"""Get the argument names used by this class.

        Args:
            **kwargs: Additional keyword arguments are passed to
                _arguments.argument_names.

        Returns:
            list: Argument names.

        """
        return cls._arguments.argument_names(**kwargs)

    @classmethod
    def ignored(cls, name):
        r"""Check if an argument is ignored.

        Args:
            name (str): Argument name.

        Returns:
            bool: True if the argument is ignored, False otherwise.

        """
        return cls._arguments[name].ignored

    @classmethod
    def parse(cls, x, args, name=None, **kwargs):
        r"""Parse an argument.

        Args:
            x (object): Instance to parse.
            args (argparse.Namespace): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to the
                from_args method.

        Returns:
            object: The parsed instance.

        """
        assert name is None
        return cls._arguments.parse(x, args, **kwargs)

    # def output_suffix(self, args, output, wildcards=None):
    #     r"""Generate an output suffix from the arguments.

    #     Args:
    #         args (argparse.Namespace): Parsed arguments.
    #         output (str): Name of output to generate suffix for.
    #         wildcards (list, optional): List of arguments that wildcards
    #             should be used for in the generated output file name.

    #     Returns:
    #         object: Argument value.

    #     """
    #     if hasattr(self, '_output_suffix'):
    #         return self._output_suffix(args, output, wildcards=wildcards)
    #     generator = self._arguments.output_suffix
    #     if generator is None:
    #         if wildcards and self._registry_key in wildcards:
    #             return '*'
    #         return self._name
    #     return self._arguments.output_suffix(
    #         args, output, wildcards=wildcards)

    @property
    def value(self):
        r"""object: Parsed base argument value."""
        return self._name

    @property
    def string(self):
        r"""str: String representation of this variable."""
        out = self.value
        if isinstance(out, str):
            return out
        return None

    def string_glob(self, wildcards):
        r"""Create a string with the provided list of parameters replaced
        with *.

        Args:
            wildcards (list): Arguments that should be replaced.

        Returns:
            str: Glob pattern.

        """
        if self._registry_key in wildcards:
            return '*'
        return self.string


class RegisteredArgumentClassKwargsBase(RegisteredArgumentClassBase):
    r"""Base class that is initialized by a set of keyword arguments.

    Args:
        **kwargs: Keyword arguments are treated as arguments.

    """

    def __init__(self, **kwargs):
        args = self._arguments.kwargs2args(kwargs)
        super(RegisteredArgumentClassKwargsBase, self).__init__(args)

    @classmethod
    def from_args(cls, args, **kwargs):
        r"""Construct the class from a argument namespace.

        Args:
            args (argparse.Namespace): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to the
                class constructor.

        Returns:
            cls: Instance of class.

        """
        kwargs.update(
            **cls._arguments.args2kwargs(
                args, include=cls._arguments.argument_names()))
        return cls(**kwargs)


class RegisteredArgumentClassDict(RegisteredArgumentClassBase):
    r"""Base class with arguments stored in a dict called args.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Attributes:
        args (dict): Set of parsed arguments.

    """

    _args_type = 'dict'


class RegisteredArgumentClassAttributes(RegisteredArgumentClassKwargsBase):
    r"""Base class with arguments assigned as attributes on the class.

    Args:
        **kwargs: Keyword arguments are treated as arguments.

    """

    _args_type = 'attributes'
