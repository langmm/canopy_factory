import abc
import copy
import argparse
import inspect
import uuid
import numpy as np
from collections import OrderedDict
from canopy_factory import utils
from canopy_factory.utils import (
    units, cached_property, get_class_registry,
    RegisteredClassBase, NoDefault,
    SuffixGenerationError,
)


class MissingSourceError(BaseException):
    r"""Error to raise when source does not exist."""

    def __init__(self, name, value=True):
        self.value = value
        super(MissingSourceError, self).__init__(name)


class SuffixGenerator(object):
    r"""Class for generating a suffix from an argument.

    Args:
        arg (ArgumentDescription, optional): Argument description.
        value (object, optional): Value to use in the suffix when
            the argument is set.
        prefix (str, optional): Prefix to use with value.
        suffix (str, optional): Suffix to use with value.
        cond (bool, optional): Condition under which the argument
            is considered set. If not provided, the boolean value
            of the argument will be used.
        noteq (object, optional): Set the cond to when the value
            is not equal to this.
        default (object, optional): Value to use when cond is False.
        title (bool, optional): If True, use title case for the
            value.
        conv (callable, optional): Function that should be used to
            convert the argument value to a string.
        sep (str, optional): Separator to use between list/array
            arguments.
        outputs (list, optional): Outputs that this suffix is valid for.
        skip_outputs (list, optional): Outputs that this suffix is not
            valid for.
        require_all (bool, optional): If True, require that all arguments
            in a set have values.
        disabled (bool, optional): If True, the suffix is disabled.
        index (int, optional): Index that should be used for ordering
            suffixes.

    """

    def __init__(self, arg=NoDefault, value=NoDefault, prefix=NoDefault,
                 suffix=NoDefault, cond=NoDefault, noteq=NoDefault,
                 default=NoDefault, title=NoDefault, conv=NoDefault,
                 sep='_', outputs=NoDefault,
                 skip_outputs=NoDefault, require_all=False,
                 disabled=False, index=0):
        if prefix is NoDefault:
            prefix = ''
        if suffix is NoDefault:
            suffix = ''
        if title is NoDefault:
            title = False
        if isinstance(outputs, str):
            outputs = [outputs]
        if isinstance(skip_outputs, str):
            skip_outputs = [skip_outputs]
        assert not (cond is not NoDefault and noteq is not NoDefault)
        self.arg = arg
        self.value = value
        self.prefix = prefix
        self.suffix = suffix
        self.cond = cond
        self.noteq = noteq
        self.default = default
        self.title = title
        self.conv = conv
        self.sep = sep
        self.outputs = outputs
        self.skip_outputs = skip_outputs
        self.require_all = require_all
        self.disabled = disabled
        self.index = index
        super(SuffixGenerator, self).__init__()

    def is_valid(self, output):
        r"""Check if the suffix is valid for the provided output(s).

        Args:
            output (str, list): Name of output(s) to check.

        Returns:
            bool: True if the suffix is valid, False otherwise.

        """
        # TODO: For outputs that have a base, don't include suffix
        #     parameters from base
        if self.disabled:
            return False
        if output is NoDefault:
            return True
        if isinstance(output, list):
            return any(self.is_valid(x) for x in output)
        if self.outputs is not NoDefault and output not in self.outputs:
            return False
        if ((self.skip_outputs is not NoDefault
             and output in self.skip_outputs)):
            return False
        return True

    def _value2str(self, x):
        if isinstance(x, str):
            return x
        if ((isinstance(x, (list, np.ndarray, units.QuantityArray))
             and self.sep is not NoDefault
             and not (isinstance(x, units.Quantity)))):
            return self.sep.join([self._value2str(xx) for xx in x])
        # Allow precision?
        # TODO: Handle arithmetic operators
        if isinstance(x, (float, units.Quantity)):
            return str(x).replace('.', 'p').replace(' ', '')
        return str(x)

    def _callable(self, x):
        return (x is not NoDefault and callable(x))

    def _resolve_callable(self, x, args):
        if not self._callable(x):
            return x
        return x(args)

    def eval_condition(self, value):
        r"""Evaluate the condition that determines if the suffix should
        be generated or default (if provided) should be used.

        Args:
            value (object): Argument value.

        Returns:
            bool: Value of the evaluated condition. If True, the suffix
                should be generated from the argument value. If False
                and a default is provided, the default should be used.
                If False and a default is not provided, an empty suffix
                should be returned.

        """
        if self.noteq is not NoDefault:
            if self._callable(self.noteq):

                def cond(args):
                    return (value != self.noteq(args))

            else:
                cond = (value != self.noteq)
        else:
            cond = self.cond
        if cond is NoDefault:
            try:
                cond = bool(value)
            except ValueError:
                cond = bool(len(value))
        return cond

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(self, args, output, wildcards=None, skipped=None,
                 value=NoDefault, **kwargs):
        r"""Generate the suffix.

        Args:
            args (argparse.Namespace): Parsed arguments.
            output (str): Name of output to generate suffix for.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.
            skipped (list, optional): List of arguments that should be
                skipped in the generated output file name.
            value (object, optional): Value that should be used instead
                of the args attribute.
            **kwargs: Additional keyword arguments are passed to the
                suffix generator's generate method.

        Returns:
            str: Generated suffix.

        """
        if skipped is None:
            skipped = []
        if (not self.is_valid(output)) or self.arg.dest in skipped:
            return ''
        if wildcards is None:
            wildcards = []
        if self.arg.dest in wildcards:
            return '*'
        if value is NoDefault:
            value = self.arg.from_args_for_suffix(
                args, output, wildcards=wildcards, skipped=skipped,
            )
        return self.value2suffix(
            value, args, wildcards=wildcards, **kwargs)

    def value2suffix(self, value, args, wildcards=None, force=False):
        r"""Generate the suffix string for this argument by inspecting
        args.

        Args:
            value (object): Argument value.
            args (argparse.Namespace): Parsed arguments.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.
            force (bool, optional): If True, force the condition to be
                True.

        Returns:
            str: Generated suffix.

        """
        if self.disabled:
            return ''
        if wildcards is None:
            wildcards = []
        suffix_value = self._resolve_callable(self.value, args)
        if suffix_value is NoDefault:
            suffix_value = value
        cond = True if force else self.eval_condition(value)
        cond = self._resolve_callable(cond, args)
        if not cond:
            if self.default is NoDefault:
                return ''
            suffix_value = self.default
        if self.conv is not NoDefault:
            suffix_value = self.conv(suffix_value)
        suffix_value = self._value2str(suffix_value)
        if ((any(x in suffix_value for x in ',:[](){};\"\'<>/+ ')
             or ((not wildcards) and '*' in suffix_value))):
            raise SuffixGenerationError(
                f'{self.arg.dest} suffix contains invalid characters: '
                f'{suffix_value}'
            )
        if self.title:
            suffix_value = suffix_value.title().replace('_', '')
        out = f'{self.prefix}{suffix_value}{self.suffix}'
        if wildcards:
            while '**' in out:
                out = out.replace('**', '*')
        return out

    def depends(self, output):
        r"""Determine what arguments this suffix depends on for the
        provided output.

        Args:
            output (str): Name of output to check.

        Returns:
            list: Names of arguments that the suffix depends on.

        """
        if not self.is_valid(output):
            return []
        return [
            v.dest for v in self.arg.flatten()
            if (v.dest and v.suffix_generator.is_valid(output))
        ]


class ArgumentDescriptionABC(abc.ABC):
    r"""Abstract base class providing argument mixin methods.

    Args:
        name (str): Argument name.
        ignored (bool, optional): If True, the argument will be set to
            None during adjustment.
        no_cli (bool, optional): If True, the argument will not be added
            to parsers.
        no_dest (bool, optional): If True, the argument does not have a
            destination.
        suffix_param (dict, optional): Parameters used to turn the
            argument into a file suffix.
        **kwargs: Additional keyword arguments are passed to the parent
            class.

    """
    _properties_attributes = [
        'ignored', 'no_cli', 'no_dest', 'suffix_param',
    ]
    _properties_lists = [
        'suffix_outputs', 'suffix_skip_outputs',
    ]
    _properties_dicts = [
        'suffix_param',
    ]
    _properties_index = [
        'suffix_index',
    ]
    _properties_inherit = [
        'ignored', 'no_cli',
    ]
    _modifies_name = [
        'name', 'subparser', 'subparser_specific_dest',
        'strip_classes', 'add_class', 'remove_class',
    ]

    def __init__(self, name, ignored=False, no_cli=False,
                 no_dest=False, suffix_param=NoDefault, **kwargs):
        self._name = name
        self._classes = OrderedDict()
        self.ignored = ignored
        self.no_cli = no_cli
        self.no_dest = no_dest
        self.suffix_param = suffix_param
        self.disabled = False
        super(ArgumentDescriptionABC, self).__init__(**kwargs)

    def __repr__(self):
        type_str = str(type(self)).split("'")[1]
        out = f'{type_str}('
        if self.name:
            out += self.name
        if self.dest != self.name:
            out += f', dest={self.dest}'
        out += ')'
        return out

    @property
    def name(self):
        r"""str: Name of the argument."""
        return self._name

    @property
    def dest(self):
        r"""str: Name of variable where the argument will be stored."""
        if self.no_dest:
            return None
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

    def matches(self, solf):
        r"""Check if this argument matches another in terms of the
        destination(s).

        Args:
            solf (ArgumentDescription): Argument to compare this one to.

        Returns:
            bool: True if solf matches this argument, False otherwise.

        """
        if type(self) is not type(solf):
            return False
        if self.dest is not None:
            return (self.dest == solf.dest)
        if len(self.members) != len(solf.members):
            return False
        if len(self.members) == 0:
            return False
        return all(xself.matches(xsolf)
                   for xself, xsolf in zip(self.members, solf.members))

    def findnested(self, x, default=NoDefault, check_root=False):
        r"""Find a member argument that matches the one provided.

        Args:
            x (ArgumentDescription): Argument to find.

        Returns:
            ArgumentDescription: Matching member argument.

        """
        if check_root and self.matches(x):
            return self
        for v in self.members:
            out = v.findnested(x, None, check_root=True)
            if out is not None:
                return out
        if default is not NoDefault:
            return default
        raise KeyError(x)

    def hasnested(self, k, check_root=False):
        r"""Check if an argument is nested inside a member argument set.

        Args:
            k (str): Key name to find an argument for.
            check_root (bool, optional): If True, check the root.

        Returns:
            bool: True if the argument is a nested member, False
                otherwise.

        """
        if check_root and self.dest is not None and k == self.dest:
            return True
        for v in self.members:
            if v.hasnested(k, check_root=True):
                return True
        return False

    def getnested(self, k, default=NoDefault, check_root=False):
        r"""Retrieve an argument that may be nested inside another
        argument set.

        Args:
            k (str): Key name to find an argument for.
            default (object, optional): Value to return if an argument
                cannot be located.
            check_root (bool, optional): If True, check the root.

        Returns:
            ArgumentDescription: Argument description instance.

        """
        if check_root and self.dest is not None and k == self.dest:
            return self
        for v in self.members:
            out = v.getnested(k, None, check_root=True)
            if out is not None:
                return out
        if default is not NoDefault:
            return default
        raise KeyError(k)

    def argument_names(self, include='dest', no_root=False):
        r"""Get all fo the associated argument names.

        Args:
            include (str, optional): Specifies what to include in the
                returned list. Supported options include::

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

    def arguments_set(self, args):
        r"""Check which arguments are set.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            list: Set of arguments that are set.

        """
        out = []
        if ((self.dest and (not self.ignored)
             and getattr(args, self.dest, None) is not None)):
            out += [self.dest]
        for v in self.members:
            out += v.arguments_set(args)
        return out

    def _get_property(self, name, src=None, from_name=False):
        if name in self._properties_attributes:
            return getattr(self, name)
        if name.startswith('suffix_'):
            if self.suffix_param is NoDefault:
                raise MissingSourceError(name, self.generates_suffix)
            src = self.suffix_param
            name = name.split('suffix_', maxsplit=1)[-1]
        if src is None:
            if name not in self._properties_inherit:
                assert (from_name and self.dest
                        and self.hasnested(self.dest))
                raise MissingSourceError(name, False)
            raise MissingSourceError(name)
        return src.get(name, NoDefault)

    def _set_property(self, name, value, src=None, from_name=False):
        if name in self._properties_attributes:
            setattr(self, name, value)
            return
        if name == 'name':
            setattr(self, '_name', value)
            return
        is_suffix = False
        if name.startswith('suffix_'):
            if self.suffix_param is NoDefault:
                raise MissingSourceError(name, self.generates_suffix)
            src = self.suffix_param
            name = name.split('suffix_', maxsplit=1)[-1]
            is_suffix = True
        if src is None:
            if name not in self._properties_inherit:
                assert (from_name and self.dest
                        and self.hasnested(self.dest))
                raise MissingSourceError(name, False)
            raise MissingSourceError(name)
        src[name] = value
        if is_suffix:  # Force inheritance
            raise MissingSourceError(f'suffix_{name}')

    def _modify(self, name, value, src=None, from_name=False):
        r"""Modify an argument property.

        Args:
            name (str): Property name. If the name starts with 'append_',
                the value will be added to the existing values.
            value (object): New property value.
            src (dict, optional): Dictionary for properties in
                self._properties_dicts should be stored.
            from_name (bool, optional): If True, this modification is
                name specific.

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
        elif name == 'strip_suffix_outputs':
            if not value:
                return False
            if self.suffix_param is not NoDefault:
                self.suffix_param.pop('outputs', None)
            return True
        elif name == 'replace_suffix_outputs':
            if (not value) or (not self.generates_suffix):
                return False
            if self.suffix_param is NoDefault:
                return True
            if 'outputs' in self.suffix_param:
                for k, v in value.items():
                    if k in self.suffix_param['outputs']:
                        self.suffix_param['outputs'].remove(k)
                        self.suffix_param['outputs'] += v
            if 'skip_outputs' in self.suffix_param:
                for k, v in value.items():
                    if k in self.suffix_param['skip_outputs']:
                        self.suffix_param['skip_outputs'].remove(k)
                        self.suffix_param['skip_outputs'] += v
            return True
        elif name in ['append_suffix_outputs',
                      'append_suffix_skip_outputs']:
            if (not value) or (not self.generates_suffix):
                return False
            if self.suffix_param is NoDefault:
                return True
            outputs = self.suffix_param.get('outputs', NoDefault)
            skip_outputs = self.suffix_param.get(
                'skip_outputs', NoDefault)
            if outputs is NoDefault:
                outputs = []
            if skip_outputs is NoDefault:
                skip_outputs = []
            if name == 'append_suffix_skip_outputs':
                dest = skip_outputs
            else:
                dest = outputs
            if isinstance(value, list):
                dest += [
                    x for x in value
                    if x not in outputs + skip_outputs
                ]
            else:
                assert isinstance(value, dict)
                for k, v in value.items():
                    if k is None:
                        dest += [
                            x for x in v
                            if x not in outputs + skip_outputs
                        ]
                    elif k in skip_outputs:
                        skip_outputs += [
                            x for x in v
                            if x not in outputs + skip_outputs
                        ]
                    elif k in outputs:
                        outputs += [
                            x for x in v
                            if x not in outputs + skip_outputs
                        ]
            if not outputs:
                outputs = NoDefault
            if not skip_outputs:
                skip_outputs = NoDefault
            self.suffix_param['outputs'] = outputs
            self.suffix_param['skip_outputs'] = skip_outputs
            return True
        elif name == 'add_class':
            self.add_class(value)
            return False
        elif name == 'remove_class':
            self.remove_class(value)
            return False
        if name.startswith('append_'):
            name = name.split('append_', maxsplit=1)[-1]
            try:
                existing = self._get_property(name, src=src,
                                              from_name=from_name)
            except MissingSourceError as e:
                return e.value
            if name in self._properties_lists:
                if not isinstance(value, list):
                    value = [value]
                if existing is NoDefault:
                    existing = []
                value = list(existing) + value
            elif name in self._properties_dicts:
                assert isinstance(value, dict)
                if existing is NoDefault:
                    existing = {}
                value = dict(existing, **value)
            else:
                raise ValueError(f'\"{name}\" is not a list or dict')
        elif name.startswith('remove_'):
            name = name.split('remove_', maxsplit=1)[-1]
            try:
                existing = self._get_property(name, src=src,
                                              from_name=from_name)
            except MissingSourceError as e:
                return e.value
            if name in self._properties_lists:
                if existing is NoDefault:
                    existing = []
                value = [x for x in existing if x not in value]
            elif name in self._properties_dicts:
                if existing is NoDefault:
                    existing = {}
                value = {k: v for k, v in existing.items()
                         if k not in value}
            else:
                raise ValueError(f'\"{name}\" is not a list or dict')
        elif name.startswith('increment_'):
            name = name.split('increment_', maxsplit=1)[-1]
            if name not in self._properties_index:
                raise ValueError(f'\"{name}\" is not an index')
            assert isinstance(value, int)
            if not value:
                return False
            try:
                existing = self._get_property(name, src=src,
                                              from_name=from_name)
            except MissingSourceError as e:
                return e.value
            if existing is NoDefault:
                existing = 0
            value = existing + value
        elif name.startswith('default_'):
            name = name.split('default_', maxsplit=1)[-1]
            try:
                existing = self._get_property(name, src=src,
                                              from_name=from_name)
            except MissingSourceError as e:
                return e.value
            if existing is not NoDefault:
                return True
        if name in self._properties_lists and not isinstance(value, list):
            value = [value]
        if name == 'keys':
            value = tuple(value)
        try:
            self._set_property(name, value, src=src, from_name=from_name)
            return (name in self._properties_inherit)
        except MissingSourceError as e:
            return e.value

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
        # Perform modifications that alter name first
        kwargs_members = {}
        for k in self._modifies_name:
            if k in kwargs:
                v = kwargs.pop(k)
                if self._modify(k, v):
                    kwargs_members[k] = v
        name = self.dest
        kwargs_name = {}
        if name is not None:
            if modifications and name in modifications:
                kwargs_name.update(**modifications[name])
            if include is not None and name not in include:
                self.disabled = True
            if exclude is not None and name in exclude:
                self.disabled = True
            if ignore is not None and name in ignore:
                kwargs_name['ignored'] = True
        elif self.no_dest and self.name:
            if modifications and self.name in modifications:
                kwargs_name.update(**modifications[self.name])
        if self.disabled:
            return
        for k, v in kwargs.items():
            if self._modify(k, v):
                kwargs_members[k] = v
        for k, v in kwargs_name.items():
            if self._modify(k, v, from_name=True):
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
        if self.ignored or self.no_cli:
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

    @property
    def generates_suffix(self):
        r"""bool: True if the argument generates a suffix."""
        if self.suffix_param is not NoDefault:
            return True
        for v in self.members:
            if v.generates_suffix:
                return True
        return False

    @property
    def suffix_generator(self):
        r"""SuffixGenerator: Suffix generator."""
        if self.suffix_param is NoDefault:
            kws = {'disabled': (not self.generates_suffix)}
            return SuffixGenerator(self, **kws)
        return SuffixGenerator(self, **self.suffix_param)

    def generate_suffix(self, *args, **kwargs):
        r"""Generate the suffix string for this argument by inspecting
        args.

        Args:
            *args: Additional arguments are passed to the
                suffix generator's generate method.
            **kwargs: Additional keyword arguments are passed to the
                suffix generator's generate method.

        Returns:
            str: Generated suffix.

        """
        return self.suffix_generator.generate(*args, **kwargs)

    def from_args_for_suffix(self, args, output, wildcards=None,
                             skipped=None):
        r"""Get the argument in a form for generating the suffix.

        Args:
            args (argparse.Namespace): Parsed arguments.
            output (str): Name of output to generate suffix for.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.
            skipped (list, optional): List of arguments that should be
                skipped in the generated output file name.

        Returns:
            object: Argument value.

        """
        value = self.from_args(args)
        if isinstance(value, RegisteredArgumentClassBase):
            value = value._name
        return value

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
    _modifies_name = (
        ArgumentDescriptionABC._modifies_name + [
            'dest', 'prefix', 'suffix',
        ]
    )

    def __init__(self, keys, properties, prefix='', suffix='',
                 description='', universal=False, **kwargs):
        for k in ArgumentDescriptionABC._properties_attributes:
            if k in properties:
                assert k not in kwargs
                kwargs[k] = properties.pop(k)
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
        if self.no_dest:
            assert 'dest' not in self.properties
            assert not (self.prefix or self.suffix)
            return None
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
        out = NoDefault
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

    def _modify(self, name, value, src=None, **kwargs):
        if name == 'subparser':
            if self.subparser_specific_dest is True:
                self.subparser_specific_dest = value
            return True
        if ((name == 'class_converter'
             and value != self._class_converter_method)):
            self._class_converter_method = None
            return False
        if src is None:
            src = self.properties
        return super(ArgumentDescription, self)._modify(
            name, value, src=src, **kwargs)

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
        if only_subparsers or self.ignored or self.no_cli:
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

    def from_args(self, args, default=NoDefault):
        r"""Construct the argument from an argument namespace.

        Args:
            args (argparse.Namespace): Parsed arguments.
            default (object, optional): Alternate default to use.

        Returns:
            object: Argument value.

        """
        if self.ignored:
            return None
        out = getattr(args, self.dest, None)
        if out is None:
            if default is NoDefault:
                default = self.default
            if default is not NoDefault:
                out = default
        return self.finalize(out, args=args)

    def finalize(self, x, args=None, **kwargs):
        r"""Finalize an argument.

        Args:
            x (object): Argument value to finalize.
            args (argparse.Namespace, optional): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to the
                base method.

        Returns:
            object: The finalized instance.

        """
        if 'type' in self.properties and x is not None:
            if self.properties.get('action', None) in ['append', 'extend']:
                assert isinstance(x, list)
                x = [self.properties['type'](xx) for xx in x]
            else:
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
        return super(ArgumentDescription, self).finalize(
            x, args=args, **kwargs)

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
        **kwargs: Additional keyword arguments are passed to the parent
            class.

    """

    _properties_attributes = (
        ArgumentDescription._properties_attributes + [
            'using_flag',
        ]
    )

    def __init__(self, name, properties, subparser_properties=None,
                 subparser_arguments=None, using_flag=False, **kwargs):
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
            keys, properties, **kwargs)
        if self.generates_suffix and self.suffix_param is NoDefault:
            self.suffix_param = {}

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
        if self.ignored or self.no_cli:
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

    def from_args_for_suffix(self, args, output, **kwargs):
        r"""Get the argument in a form for generating the suffix.

        Args:
            args (argparse.Namespace): Parsed arguments.
            output (str): Name of output to generate suffix for.
            **kwargs: Additional keyword arguments are passed to the
                base class and calls to suffix generate for the selected
                subparser.

        Returns:
            object: Argument value.

        """
        # subparser = super(SubparserArgumentDescription, self).from_args(
        #     args)
        subparser = super(
            SubparserArgumentDescription, self).from_args_for_suffix(
                args, output, **kwargs)
        assert isinstance(subparser, str)
        assert subparser in self.subparser_arguments
        value = [
            self.suffix_generator.generate(
                args, output, value=subparser, **kwargs)
        ]
        value_sub = (
            self.subparser_arguments[subparser].generate_suffix(
                args, output, **kwargs)
        )
        if isinstance(value_sub, list):
            value += value_sub
        else:
            value.append(value_sub)
        return [x for x in value if x]

    def from_args(self, args, **kwargs):
        r"""Construct the argument from an argument namespace.

        Args:
            args (argparse.Namespace): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to subparser
                from_args method.

        Returns:
            object: Argument value.

        """
        subparser = super(SubparserArgumentDescription, self).from_args(
            args, default=kwargs.pop('default', NoDefault))
        if ((subparser not in self.subparser_arguments
             or (not self.subparser_arguments[subparser].cls)
             or self.subparser_arguments[subparser].dont_create)):
            return subparser
        return self.subparser_arguments[subparser].from_args(args, **kwargs)

    def prepend(self, x, dont_copy=False, **kwargs):
        r"""Prepend one or more arguments to each subparser argument set.

        Args:
            x (ArgumentDescription, tuple, list): Argument to add.
            dont_copy (bool, optional): If True, don't copy the argument
                when it is added.
            **kwargs: Additional keyword arguments will be used to
                modify a copy of x prior to adding it.

        """
        for v in self.members:
            v.prepend(x, dont_copy=dont_copy, **kwargs)

    def append(self, x, dont_copy=False, **kwargs):
        r"""Append one or more arguments to each subparser argument set.

        Args:
            x (ArgumentDescription, tuple, list): Argument to add.
            dont_copy (bool, optional): If True, don't copy the argument
                when it is added.
            **kwargs: Additional keyword arguments will be used to
                modify a copy of x prior to adding it.

        """
        for v in self.members:
            v.append(x, dont_copy=dont_copy, **kwargs)


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
        **kwargs: Additional keyword arguments are passed to the parent
            class.

    """

    def __init__(self, name, properties=None, subparser_properties=None,
                 subparser_arguments=None, modifications=None,
                 include=None, exclude=None, dont_create=False,
                 **kwargs):
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
            subparser_arguments=subparser_arguments, **kwargs)
        if include is not None:
            include = [self.dest] + include
        self.modify(modifications=modifications,
                    include=include, exclude=exclude)


class ArgumentDescriptionSet(ArgumentDescriptionABC, utils.SimpleWrapper):
    r"""A set of CLI arguments.

    Args:
        arguments (list, optional): Argument descriptions.
        name (str, optional): Name to give the argument set.
        no_dest (bool, optional): If True, the argument does not have a
            destination. Defaults to True for sets.
        dont_create (bool, optional): If True, don't create an instance
            of the argument class containing this argument set during
            calls to from_args.
        arg_kwargs (dict, optional): Arguments to provided to each
            argument.
        suffix_param (dict, optional): Parameters used to turn the
            argument set into a file suffix.
        prefix (str, optional): Prefix that should be added to the
            names of all arguments in the set.
        suffix (str, optional): Suffix that should be added to the
            names of all arguments in the set.
        description (str, optional): Description string that should be
            used to format the help messages for arguments in the set.
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
    _properties_inherit = (
        ArgumentDescriptionABC._properties_inherit + [
            'subparser', 'subparser_specific_dest',
            'strip_classes',
            'prefix', 'suffix', 'description', 'universal',
        ]
    )

    def __init__(self, arguments=None, name=None, no_dest=True,
                 dont_create=False, arg_kwargs=NoDefault,
                 suffix_param=NoDefault, **kwargs):
        if arguments is None:
            arguments = []
        if arg_kwargs is NoDefault:
            arg_kwargs = {}
        for k in ['prefix', 'suffix', 'description', 'ignore']:
            if k in kwargs:
                assert k not in arg_kwargs
                arg_kwargs[k] = kwargs.pop(k)
        self.cls_kwargs = kwargs
        self.dont_create = dont_create
        super(ArgumentDescriptionSet, self).__init__(
            name, no_dest=no_dest, ordered=True,
            suffix_param=suffix_param)
        if not isinstance(arguments, list):
            arguments = [arguments]
        for x in arguments:
            self.append(x, **arg_kwargs)
        if self.generates_suffix and self.suffix_param is NoDefault:
            self.suffix_param = {}

    def __repr__(self):
        out = super(ArgumentDescriptionSet, self).__repr__()[:-1]
        args = ', '.join([f'{k}: {repr(v)}' for k, v in self.items()])
        if out[-1] != '(':
            out += ', '
        out += f'[{args}])'
        return out

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
            strip_classes (bool, optional: If True, remove the class
                associated with this set. If 'all', remove the classes
                associated with this set and all of it's members.
            **kwargs: Additional keyword arguments are used to update
                individual argument properties.

        """
        super(ArgumentDescriptionSet, self).modify(
            modifications=modifications, include=include,
            exclude=exclude, **kwargs)
        remove = [
            k for k, v in self.items()
            if (v.disabled
                or (v.dest is None
                    and isinstance(v, ArgumentDescriptionSet)
                    and len(v) == 0))
        ]
        for k in remove:
            del self[k]
        if any(k in kwargs for k in self._modifies_name):
            self.reset_keys()

    def from_args_for_suffix(self, args, output, **kwargs):
        r"""Get the argument in a form for generating the suffix.

        Args:
            args (argparse.Namespace): Parsed arguments.
            output (str): Name of output to generate suffix for.
            **kwargs: Additional keyword arguments are passed to the
                suffix generators for members.

        Returns:
            object: Argument value.

        """
        values = OrderedDict()
        for k, v in self.items():
            if not v.generates_suffix:
                continue
            values[k] = v.generate_suffix(args, output, **kwargs)
        if not any(values.values()):
            return ''
        if not self.suffix_param.get('require_all', False):
            return [v for v in values.values() if v]
        if any(values.values()) and not all(values.values()):
            for k, v in self.items():
                if values[k]:
                    continue
                values[k] = v.generate_suffix(
                    args, output, force=True, **kwargs)
        return list(values.values())

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

    @property
    def unsplitable(self):
        r"""bool: True if the set cannot be split."""
        # TODO: Check other arguments?
        return (self.name is not None
                or self.suffix_param is not NoDefault)

    def dest2key(self, dest):
        r"""Convert an argument destination name to the key used in this
        set.

        Args:
            dest (str): Argument destination name.

        Returns:
            str: Argument key.

        """
        return self.getdest(dest).name

    def key2dest(self, key):
        r"""Convert a the key for an argument in this set to the
        destination name.

        Args:
            key (str): Argument key.

        Returns:
            str: Argument destination name.

        """
        return self[key].dest

    def sort_by_suffix(self):
        r"""Sort the arguments by suffix."""
        if not self.generates_suffix:
            return

        def suffix_index(x):
            k, v = x[:]
            if not v.generates_suffix:
                return 10000000000000
            assert v.suffix_param is not NoDefault
            return v.suffix_param.get('index', 0)

        items = sorted([(k, v) for k, v in self.items()], key=suffix_index)
        self.clear()
        for k, v in items:
            if v.name is None:
                self.__setitem__(k, v, dont_add_class=True)
            else:
                self.__setitem__(v.name, v, dont_add_class=True)

    def move(self, key, pos):
        r"""Move an argument to the designated position.

        Args:
            key (str): Argument key.
            pos (int): Position that the argument should be moved to.

        """
        assert pos < len(self) and pos >= 0
        existing = list(self.keys())[pos:]
        self.move_to_end(key)
        for k in existing:
            if k == key:
                continue
            self.move_to_end(k)

    def index(self, key):
        r"""Find the index of an argument in this set.

        Args:
            key (str): Argument key.

        Returns:
            int: Index of key.

        """
        return list(self.keys()).index(key)

    def insert(self, pos, x, dont_copy=False, **kwargs):
        r"""Insert one or more arguments into the argument set.

        Args:
            pos (int): Position at which the new argument(s) should be
                inserted.
            x (ArgumentDescription, tuple, list): Argument to add.
            dont_copy (bool, optional): If True, don't copy the argument
                when it is appended.
            **kwargs: Additional keyword arguments will be used to
                modify a copy of x prior to adding it.

        """
        existing = list(self.keys())[pos:]
        self.append(x, dont_copy=dont_copy, **kwargs)
        for k in existing:
            self.move_to_end(k)

    def prepend(self, x, dont_copy=False, **kwargs):
        r"""Add an argument to the argument set at the beginning.

        Args:
            x (ArgumentDescription, tuple, list): Argument to add.
            dont_copy (bool, optional): If True, don't copy the argument
                when it is added.
            **kwargs: Additional keyword arguments will be used to
                modify a copy of x prior to adding it.

        """
        existing = list(self.keys())
        self.append(x, dont_copy=dont_copy, **kwargs)
        for k in existing:
            self.move_to_end(k)

    def append(self, x, dont_copy=False, **kwargs):
        r"""Add an argument to the argument set.

        Args:
            x (ArgumentDescription, tuple, list): Argument to add.
            dont_copy (bool, optional): If True, don't copy the argument
                when it is added.
            **kwargs: Additional keyword arguments will be used to
                modify a copy of x prior to adding it.

        """
        if isinstance(x, ArgumentDescriptionSet) and not x.unsplitable:
            if not dont_copy:
                kwargs.setdefault('strip_classes', True)
            for v in x.values():
                self.append(v, dont_copy=dont_copy, **kwargs)
            return
        elif isinstance(x, ArgumentDescriptionABC):
            if not dont_copy:
                x = x.copy(**kwargs)
            elif kwargs:
                x.modify(**kwargs)
        elif isinstance(x, tuple):
            x = ArgumentDescription(*x)
            if kwargs:
                x.modify(**kwargs)
        elif isinstance(x, list):
            for xx in x:
                self.append(xx, dont_copy=dont_copy, **kwargs)
            return
        key = x.name
        if key is None:
            key = str(uuid.uuid4())
        self[key] = x

    def __setitem__(self, k, v, dont_add_class=False):
        vnested = self.getnested(k, None)
        if vnested is not None:
            if vnested == v:
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

    def getkey(self, k, default=NoDefault):
        r"""Retrieve an argument based on a key value.

        Args:
            k (str): Key name to find an argument for.
            default (object, optional): Value to return if an argument
                cannot be located.

        Returns:
            ArgumentDescription: Argument description instance.

        """
        for v in self.flatten():
            if isinstance(v, ArgumentDescription) and k in v.keys:
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
        items = list(self.items())
        self.clear()
        for k, v in items:
            if v.name is None:
                self.__setitem__(k, v, dont_add_class=True)
            else:
                self.__setitem__(v.name, v, dont_add_class=True)


class DimensionArgumentDescription(ArgumentDescriptionSet):
    r"""A set of arguments that specifies dimensions."""

    def __init__(self, *args, **kwargs):
        super(DimensionArgumentDescription, self).__init__(
            *args, **kwargs)
        if self.generates_suffix:
            for v in self.values():
                if v.suffix_param is NoDefault:
                    v.suffix_param = {}
            self.suffix_param.setdefault('sep', 'x')
            self.suffix_param.setdefault('require_all', True)


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
                 arguments=None, **kwargs):
        if composite_type is None:
            composite_type = name
        name = prefix + name + suffix
        if isinstance(composite_type, type):
            cls = composite_type
        else:
            cls_base = get_class_registry().get(
                'argument', composite_type)
            kwargs_cls = {
                k: kwargs.pop(k) for k in [
                    'description', 'ignore', 'registry_name'
                ] if k in kwargs
            }
            cls = cls_base.class_factory(name, **kwargs_cls)
        if arguments is None:
            arguments = cls._arguments
        super(CompositeArgumentDescription, self).__init__(
            arguments, name=name, no_dest=False, **kwargs)
        self.add_class(cls, overwrite=True)

    def from_args_for_suffix(self, args, output, wildcards=None,
                             skipped=None):
        r"""Get the argument in a form for generating the suffix.

        Args:
            args (argparse.Namespace): Parsed arguments.
            output (str): Name of output to generate suffix for.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.
            skipped (list, optional): List of arguments that should be
                skipped in the generated output file name.

        Returns:
            object: Argument value.

        """
        if skipped and any(v.dest in skipped for v in self.values()):
            return ''
        if wildcards and any(v.dest in wildcards for v in self.values()):
            return '*'
        value = self.from_args(args)
        out = value.string
        if out is None:
            return ''
        return out

    @property
    def is_output(self):
        r"""bool: True if the argument describes an output."""
        return (self.cls._name == 'output')

    def argument_names(self, include='dest'):
        r"""Get all fo the associated argument names.

        Args:
            include (str, optional): Specifies what to include in the
                returned list. Supported options include::

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
        cls._output_suffix = cls._arguments.suffix_generator

    @staticmethod
    def _build_arguments(cls):
        if isinstance(cls._arguments, list):
            cls._arguments = ArgumentDescriptionSet(
                cls._arguments, suffix_param={})
        cls._arguments = cls._arguments.copy(
            strip_classes=True,
            default_suffix_index=0,
        )
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

    def setarg(self, name, value):
        r"""Set an argument value.

        Args:
            name (str): Argument name.
            value (object): Argument value.

        """
        if self._args_type == 'namespace':
            setattr(self.args, name, value)
        elif self._args_type == 'dict':
            self.args[name] = value
        elif self._args_type == 'attributes':
            setattr(self, name, value)
        self.clear_cached_properties()

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

    @property
    def value(self):
        r"""object: Parsed base argument value."""
        return self._name

    @property
    def string(self):
        r"""str: String representation of this variable."""
        # out = self.value
        # if isinstance(out, str):
        #     return out
        return None


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
