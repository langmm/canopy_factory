import os
import sys
import pdb
import copy
import argparse
import numpy as np
from collections import OrderedDict
from datetime import datetime, timedelta
from canopy_factory import utils
from canopy_factory.utils import (
    parse_quantity,
    get_class_registry, NoDefault, RegisteredClassBase,
    cached_property, cached_factory_property,
)


class InstrumentedParser(argparse.ArgumentParser):
    r"""Class for parsing arguments allowing arguments to be
    added to multiple subparsers."""

    def __init__(self, *args, **kwargs):
        self._quantity_units = {}
        self._subparser_mutually_exclusive_groups = {}
        self._subparser_classes = {}
        self._subparser_defaults = {}
        self._subparser_dependencies = {}
        self._subparser_func = kwargs.pop('func', None)
        super(InstrumentedParser, self).__init__(*args, **kwargs)

    @property
    def _primary_subparser(self):
        if len(self._subparser_classes) == 1:
            return list(self._subparser_classes.keys())[0]
        return None

    def parse_args(self, args=None, **kwargs):
        r"""Parse arguments.

        Args:
            args (list, optional): Arguments to parse. Defaults to
                sys.argv if not provided.
            **kwargs: Additional keyword arguments are passed to the
                parent class's method.

        Returns:
            argparse.Namespace: Parsed arguments.

        """
        args0, unparsed = self.parse_known_args()
        args_supp = []
        for k, v in self._subparser_defaults.items():
            if v is None or getattr(args0, k, None):
                continue
            args_supp.append(v)
        if args_supp:
            if args is None:
                args = sys.argv
            args = args_supp + args[1:]
        out = super(InstrumentedParser, self).parse_args(
            args=args, **kwargs)
        out = self.add_units(out)
        return out

    def add_units(self, args):
        r"""Add units to arguments that units were recorded for.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            argparse.Namespace: Parsed arguments.

        """
        for k, v in self._quantity_units.items():
            if hasattr(args, k):
                setattr(args, k, parse_quantity(getattr(args, k), v))
        for k, v in self._subparser_classes.items():
            for vv in v.choices.values():
                vv.add_units(args)
        return args

    def has_subparsers(self, name):
        r"""Check if there is a subparsers group associated with the
        specified name.

        Args:
            name (str): Name of the subparsers group to check for.

        Returns:
            bool: True if the named subparsers group exists.

        """
        return (name in self._subparser_classes)

    def get_subparsers(self, name, default=NoDefault):
        r"""Get the subparsers object with the specified name.

        Args:
            name (str): Name of the subparsers instance to retrieve.
            default (object, optional): Default to return if the
                subparsers instance does not exist.

        Returns:
            InstrumentedSubparsers: Subparsers instance.

        Raises:
            KeyError: If default not provided and the subparsers object
                does not exist.

        """
        out = default
        if name in self._subparser_classes:
            out = InstrumentedSubparsers(
                name, self._subparser_classes[name], self)
        if out is NoDefault:
            raise KeyError(name)
        return out

    def add_subparsers(self, *args, **kwargs):
        r"""Add a subparsers group to this parser.

        Args:
            *args, **kwargs: Arguments are passed to the base class's
                method.

        Returns:
            InstrumentedSubparsers: Wrapped subparsers group.

        """
        name = kwargs['dest']
        self._subparser_dependencies[name] = {}
        self._subparser_defaults[name] = kwargs.pop('default', None)
        kwargs.setdefault('parser_class', InstrumentedParser)
        out = super(InstrumentedParser, self).add_subparsers(
            *args, **kwargs)
        self._subparser_classes[name] = out
        return InstrumentedSubparsers(name, out, self)

    def run_subparser(self, name, args):
        r"""Run the subparser selected by a subparser group.

        Args:
            name (str): Name of the subparser group to run.
            args (argparse.Namespace): Parsed arguments.

        """
        subparser = getattr(args, name)
        if subparser is None:
            subparser = self._subparser_defaults[name]
        func = self._subparser_classes[name].choices[
            subparser]._subparser_func
        if func is None:
            func = get_class_registry().get(name, subparser)
        print(f"RUNNING {subparser}")
        func(args)

    def _handle_local_options(self, args, kwargs):
        if 'units' in kwargs:
            kwargs = copy.deepcopy(kwargs)
            self._quantity_units[kwargs['dest']] = kwargs.pop('units')
        return args, kwargs

    def _iter_subparsers(self, args, kwargs, include_name=False):
        subparsers = kwargs.pop('subparsers', None)
        subparser_options = kwargs.pop('subparser_options', {})
        subparser_specific_dest = kwargs.pop(
            'subparser_specific_dest', False)
        kwargs.setdefault('dest', args[0].lstrip('-').replace('-', '_'))
        if subparsers is False:
            assert not subparser_specific_dest
            args, kwargs = self._handle_local_options(args, kwargs)
            if include_name:
                yield None, super(InstrumentedParser, self), args, kwargs
            else:
                yield super(InstrumentedParser, self), args, kwargs
            return
        if (not isinstance(subparsers, dict)) and self._primary_subparser:
            subparsers = {self._primary_subparser: subparsers}
        if not subparsers:
            args, kwargs = self._handle_local_options(args, kwargs)
            if include_name:
                yield None, super(InstrumentedParser, self), args, kwargs
            else:
                yield super(InstrumentedParser, self), args, kwargs
            return
        if not subparsers:
            print(self._primary_subparser)
            pdb.set_trace()
        for k, v in subparsers.items():
            if v is None:
                unique_prog = []
                v = []
                if k in self._subparser_classes:
                    for kk, vv in self._subparser_classes[k].choices.items():
                        if vv.prog not in unique_prog:
                            unique_prog.append(vv.prog)
                            v.append(kk)
                if not v:
                    raise ValueError(f"Could not locate subparser or "
                                     f"group for \"{k}\"")
            for x in v:
                iargs = copy.deepcopy(args)
                ikwargs = dict(kwargs, **subparser_options.get(x, {}))
                if subparser_specific_dest:
                    ikwargs['dest'] += f'_{x}'
                    iargs = tuple(list(iargs) + [args[0] + f'-{x}'])
                if x in self._subparser_mutually_exclusive_groups:
                    xinst = self._subparser_mutually_exclusive_groups[x]
                else:
                    xinst = self._subparser_classes[k].choices[x]
                if include_name:
                    yield (x, xinst, iargs, ikwargs)
                else:
                    yield (xinst, iargs, ikwargs)

    def add_argument(self, *args, **kwargs):
        r"""Add an argument to the parser.

        Args:
            *args, **kwargs: All arguments are passed to the add_argument
                method of the subparsers (if there are any) or the parent
                class (if there are not any subparsers).

        """
        for x, iargs, ikwargs in self._iter_subparsers(args, kwargs):
            x.add_argument(*iargs, **ikwargs)

    def add_mutually_exclusive_group(self, name, *args, **kwargs):
        r"""Add a mutually exclusive group of arguments.

        Args:
            name (str): Name used to identify the mutually exclusive
                group.
            *args, **kwargs: Additional arguments are passed to the
                add_mutually_exclusive_group method of the subparsers
                (if there are any) or the parent class (if there are not
                any subparsers).

        Returns:
            InstrumentedParserGroup: Wrapped exclusive argument group.

        """
        group = {}
        for k, x, iargs, ikwargs in self._iter_subparsers(
                args, kwargs, include_name=True):
            if isinstance(x, InstrumentedParser):
                group[k] = x.add_mutually_exclusive_group(name)
            else:
                group[k] = x.add_mutually_exclusive_group()
        out = InstrumentedParserGroup(group, parent=self)
        self._subparser_mutually_exclusive_groups[name] = out
        return out


class InstrumentedSubparsers(object):
    r"""Light wrapper for a group of subparsers.

    Args:
        name (str): ID string for the group.
        subparsers (ArgumentParser): Group parser for subparsers.
        parent (ArgumentParser): Parser used to create the subparser
            group.

    """

    def __init__(self, name, subparsers, parent):
        self.name = name
        self._subparsers = subparsers
        self._parent = parent

    def add_parser(self, name, *args, **kwargs):
        r"""Add a parser to the subparser group.

        Args:
            name (str): ID string for the parser.
            *args, **kwargs: Additional arguments are passed to the
                add_parser method for the underlying subparser group.

        Returns:
            ArgumentParser: New subparser.

        """
        self._parent._subparser_dependencies[self.name][name] = kwargs.pop(
            'dependencies', [])
        return self._subparsers.add_parser(name, *args, **kwargs)


class InstrumentedParserGroup(object):
    r"""Simple wrapper for a set of parsers.

    Args:
        members (dict): Set of parsers.
        parent (ArgumentParser, optional): Parser containing this group.

    """

    def __init__(self, members, parent=None):
        self.members = members
        self.parent = parent

    def _iter_subparsers(self, args, kwargs, include_name=False):
        subparsers = kwargs.pop('subparsers', list(self.members.keys()))
        subparser_options = kwargs.pop('subparser_options', {})
        assert isinstance(subparsers, list)
        for k, v in self.members.items():
            if k not in subparsers:
                continue
            ikw = dict(kwargs, **subparser_options.get(k, {}))
            if include_name:
                yield (k, v, args, ikw)
            else:
                yield v, args, ikw

    def add_argument(self, *args, **kwargs):
        r"""Add an argument to the parser.

        Args:
            *args, **kwargs: All arguments are passed to the add_argument
                method of the subparsers (if there are any) or the parent
                class (if there are not any subparsers).

        """
        for x, iargs, ikwargs in self._iter_subparsers(args, kwargs):
            x.add_argument(*iargs, **ikwargs)


class SubparserBase(RegisteredClassBase):
    r"""Base class for tasks associated with subparsers.

    Args:
        args (argparse.Namespace, optional): Parsed arguments. If not
            provided, additional keyword arguments are parsed to create
            args and keyword arguments that are not used by the parser
            are passed to the run method.
        **kwargs: Additional keyword arguments are passed to the run
            method.

    """

    _name = None
    _help = None
    _default = None
    _argument_sources = []
    _arguments = []
    _arguments_suffix_ignore = []
    _argument_modifications = {}
    _excluded_arguments = []
    _excluded_arguments_defaults = {}
    _external_arguments = {}

    def __init__(self, args=None, **kwargs):
        super(SubparserBase, self).__init__()
        if args is None:
            kwargs.setdefault(self._registry_key, self._name)
            args, kwargs = main(**kwargs)
        self._cached_args = []
        self.adjust_args(args)
        self.args = args
        kwargs.setdefault('dont_load_existing', True)
        self.run(**kwargs)

    @staticmethod
    def _on_registration(cls):
        RegisteredClassBase._on_registration(cls)
        if cls._registry_key is None:
            return
        import inspect
        base = inspect.getmro(cls)[1]
        base_args = base.argument_dict(use_flags=True)
        local_args = cls.argument_dict(use_flags=True)
        arguments = copy.deepcopy(base_args)
        for other_base in cls._argument_sources:
            arguments.update(
                **copy.deepcopy(other_base.argument_dict(use_flags=True)))
        subparser_specific = [
            k for k, v in base_args.items()
            if v[1].get('subparser_specific_dest', False)
        ]
        cls._excluded_arguments_defaults = dict(
            cls._excluded_arguments_defaults,
            **{
                base.arg2dest(*base_args[k]): base.arg2default(*base_args[k])
                for k in cls._excluded_arguments + subparser_specific
            }
        )
        for k in cls._excluded_arguments:
            arguments.pop(k)
        arguments.update(**local_args)
        for k, v in cls._argument_modifications.items():
            arguments[k][1].update(**v)
        for kext, mods in cls._external_arguments.items():
            ext_arguments = get_class_registry().get(
                cls._registry_key, kext).argument_dict(use_flags=True)
            for k, v in mods.items():
                arguments[k] = copy.deepcopy(ext_arguments[k])
                arguments[k][1].update(**v)
        cls._arguments = list(arguments.values())
        cls._argument_modifications = {}
        cls._excluded_arguments = []
        cls._external_arguments = {}
        cls._argument_sources = []

    @classmethod
    def add_arguments(cls, parser):
        r"""Add arguments associated with this subparser to a parser.

        Args:
            parser (InstrumentedParser): Parser that the arguments
                should be added to.

        """
        if not parser.has_subparsers(cls._registry_key):
            kws = {'title': cls._registry_key,
                   'dest': cls._registry_key}
            if cls._default is not None:
                kws['default'] = cls._default
            parser.add_subparsers(**kws)
        subparsers = parser.get_subparsers(cls._registry_key)
        subparsers.add_parser(cls._name, help=cls._help)
        for iargs, ikwargs in cls._arguments:
            ikwargs = dict(ikwargs, subparsers=[cls._name])
            parser.add_argument(*iargs, **ikwargs)

    @classmethod
    def adjust_args(cls, args):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        pass

    @classmethod
    def arg2dest(cls, iargs, ikwargs):
        r"""Determine the name that will be used to store an argument.

        Args:
            iargs (tuple): Argument args for add_argument.
            ikwargs (dict): Argument kwargs for add_argument.

        Returns:
            str: Name that argument will be stored under.

        """
        ikey = ikwargs.get(
            'dest', iargs[0].lstrip('--').replace('-', '_'))
        if ikwargs.get('subparser_specific_dest', False):
            ikey += f'_{cls._name}'
        return ikey

    @classmethod
    def arg2default(cls, iargs, ikwargs):
        r"""Determine the default value for an argument.

        Args:
            iargs (tuple): Argument args for add_argument.
            ikwargs (dict): Argument kwargs for add_argument.

        Returns:
            object: Default value for the argument.

        """
        if ikwargs.get('action', None) == 'store_true':
            ival = False
        elif ikwargs.get('action', None) == 'store_false':
            ival = True
        else:
            ival = ikwargs.get('default', None)
        return ival

    @classmethod
    def argument_dict(cls, use_flags=False):
        r"""Get a dictionary of argument data.

        Args:
            use_flags (bool, optional): If False, use the argument
                destinations. If True, use the argument flags.

        Returns:
            dict: Positional and keyword arguments for creating the
                arguments.

        """
        out = OrderedDict()
        for iargs, ikwargs in cls._arguments:
            if use_flags:
                ikey = iargs[0]
            else:
                ikey = cls.arg2dest(iargs, ikwargs)
            out[ikey] = [iargs, ikwargs]
        return out

    @classmethod
    def argument_defaults(cls, use_flags=False):
        r"""Get a dictionary of argument defaults.

        Args:
            use_flags (bool, optional): If False, use the argument
                destinations. If True, use the argument flags.

        Returns:
            dict: Argument defaults.

        """
        parser = InstrumentedParser('for defaults')
        cls.add_arguments(parser)
        args = parser.parse_args([cls._name])
        args._in_argument_defaults = True
        cls.adjust_args(args)
        out = {}
        for iargs, ikwargs in cls._arguments:
            dest = cls.arg2dest(iargs, ikwargs)
            if use_flags:
                ikey = iargs[0]
            else:
                ikey = dest
            out[ikey] = getattr(args, dest)
        return out

    @classmethod
    def argument_names(cls, use_flags=False):
        r"""Set of arguments allowed by the subparser.

        Args:
            use_flags (bool, optional): If False, use the argument
                destinations. If True, use the argument flags.

        Returns:
            list: Arguments.

        """
        out = []
        for iargs, ikwargs in cls._arguments:
            if use_flags:
                out.append(iargs[0])
            else:
                out.append(cls.arg2dest(iargs, ikwargs))
        return out

    @classmethod
    def select_valid_arguments(cls, args, use_flags=False):
        r"""Select arguments from a list that are valid for this class.

        Args:
            args (list, dict): Set of argument names.
            use_flags (bool, optional): If False, use the argument
                destinations. If True, use the argument flags.

        """
        names = cls.argument_names(use_flags=use_flags)
        return [k for k in args if k in names]

    def copy_args(self, args):
        r"""Copy arguments from the namespace to this instance as
        attributes.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        for k in self.argument_names():
            assert not hasattr(self, k)
            setattr(self, k, getattr(args, k))

    def cache_args(self, adjust=None, args_preserve=None,
                   args_overwrite=None, properties_preserve=None,
                   properties_overwrite=None, alternate_outputs=None,
                   recursive=None):
        r"""Cache the current set of arguments.

        Args:
            adjust (type, optional): Class that should be used to adjust
                the arguments after they are updated. If not provided,
                this class will be used.
            args_preserve (list, optional): Set of argument names to
                preserve from the updated argument namespace when the
                cached args are restored (by restore_args).
            args_ovewrite (dict, optional): Argument values to set for
                the run after copying the current argument namespace.
            properties_preserve (list, optional): Set of cached
                properties that should be preserved.
            properties_overwrite (dict, optional): Cached property
                values to set for the run after caching the current
                properties.
            alternate_outputs (list, optional): Set of alternate outputs
                that should be generated.
            recursive (bool, optional): If True, this is a recursive
                call and overwrite for outputs should be reset to False.
                If not provided, recursive will be set to true if self
                is an instance of the provided adjust class.

        """
        if adjust is None:
            adjust = self
        if args_preserve is None:
            args_preserve = []
        if args_overwrite is None:
            args_overwrite = {}
        if properties_preserve is None:
            properties_preserve = []
        if properties_overwrite is None:
            properties_overwrite = {}
        if alternate_outputs is None:
            alternate_outputs = []
        if recursive is None:
            recursive = (adjust._name == self._name)
        enabled_outputs = adjust.enabled_outputs(self.args)
        for name in adjust.output_names():
            output_key = f'output_{name}'
            if name in enabled_outputs + alternate_outputs:
                args_overwrite.setdefault(output_key, True)
            else:
                args_overwrite.setdefault(output_key, False)
            if recursive:
                args_overwrite.setdefault(f'overwrite_{name}', False)
        cached_properties = self.pop_cached_properties(
            preserve=properties_preserve, args=True)
        self._cached_args.append(
            (self.args, args_preserve,
             cached_properties, properties_preserve)
        )
        self.args = copy.deepcopy(self.args)
        self.log(f'Updating args with {args_overwrite}')
        for k, v in args_overwrite.items():
            setattr(self.args, k, v)
        if properties_overwrite:
            self.set_cached_properties(properties_overwrite)
        adjust.adjust_args(self.args)

    def restore_args(self):
        r"""Restore the last set of preserved arguments."""
        new_args = self.args
        cache_entry = self._cached_args.pop()
        self.args = cache_entry[0]
        for k in cache_entry[1]:
            setattr(self.args, k, getattr(new_args, k))
        cached_properties = cache_entry[2]
        props_preserve = cache_entry[3]
        if props_preserve:
            cached_properties.update(
                self.get_cached_properties(include=props_preserve)
            )
        self.reset_cached_properties(cached_properties, args=True)

    def run(self):
        r"""Run the process associated with this subparser."""
        raise NotImplementedError


class TaskBase(SubparserBase):
    r"""Base class for tasks."""

    _registry_key = 'task'
    _default = 'generate'
    _ext = None
    _output_dir = None
    _alternate_outputs_write_optional = []
    _alternate_outputs_write_required = []
    _arguments_suffix_ignore = [
        'overwrite_all', 'verbose', 'debug',
    ]
    _arguments = [
        (('--output', ), {
            'type': str,
            'help': 'File where output should be saved',
            'subparser_specific_dest': True,
        }),
        (('--overwrite', ), {
            'action': 'store_true',
            'help': 'Overwrite existing output',
            'subparser_specific_dest': True,
        }),
        (('--overwrite-all', ), {
            'action': 'store_true',
            'help': 'Overwrite all child components of the task',
        }),
        (('--verbose', ), {
            'action': 'store_true',
            'help': 'Show log messages'
        }),
        (('--debug', ), {
            'action': 'store_true',
            'help': ('Run in debug mode, setting break points for debug '
                     'messages and errors')
        }),
    ]

    def __init__(self, *args, **kwargs):
        self._alternate_output = {}
        super(TaskBase, self).__init__(*args, **kwargs)

    @staticmethod
    def _on_registration(cls):
        SubparserBase._on_registration(cls)
        if cls._registry_key is None:
            return
        import inspect
        base = inspect.getmro(cls)[1]
        cls._arguments_suffix_ignore = copy.deepcopy(
            base._arguments_suffix_ignore) + cls._arguments_suffix_ignore
        cls._alternate_outputs_write = (
            cls._alternate_outputs_write_required
            + cls._alternate_outputs_write_optional)

    def output_exists(self, name=None):
        r"""bool: True if the output file exists."""
        if isinstance(name, list):
            return all(self.output_exists(name=k) for k in name)
        if name is None:
            name = self._name
        overwrite = getattr(self.args, f'overwrite_{name}')
        output = getattr(self.args, f'output_{name}')
        return (output and (not overwrite) and os.path.isfile(output))

    @classmethod
    def run_class(cls, self, dont_load_existing=False,
                  args_preserve=None, args_overwrite=None,
                  properties_preserve=None, properties_overwrite=None,
                  recursive=None, dont_reset_alternate_output=False,
                  return_alternate_output=False,
                  require_alternate_output=None, **kwargs):
        r"""Run the process associated with this subparser.

        Args:
            self (object): Task instance that is running.
            dont_load_existing (bool, optional): If True, existing output
                will not be loaded.
            args_preserve (list, optional): Set of argument names to
                preserve from the updated arguments following a run.
            args_ovewrite (dict, optional): Argument values to set for
                the run after copying the current argument namespace.
            properties_preserve (list, optional): Set of cached
                properties that should be preserved.
            properties_overwrite (dict, optional): Cached property
                values to set for the run after caching the current
                properties.
            recursive (bool, optional): If True, this is a recursive
                call and overwrite for outputs should be reset to False.
                If not provided, recursive will be set to true if self
                is an instance of the provided adjust class.
            dont_reset_alternate_output (bool, optional): If True, don't
                reset the dictionary of alternate outputs on return.
            return_alternate_output (str, optional): Name of an
                alternate output that should be output instead of the
                standard output.
            require_alternate_output (list, optional): A list of
                alternate output values that should be recorded. If not,
                provided the default required alternate outputs will be
                used.
            **kwargs: Additional keyword arguments are passed to the
                _run method if it is called.

        Returns:
            object: Generated object.

        """
        output_name = cls._name
        if ((return_alternate_output
             and return_alternate_output in cls._alternate_outputs_write)):
            if require_alternate_output is None:
                require_alternate_output = []
            if return_alternate_output not in require_alternate_output:
                require_alternate_output.append(return_alternate_output)
            output_name = return_alternate_output
            return_alternate_output = False
        self.cache_args(adjust=cls, args_preserve=args_preserve,
                        args_overwrite=args_overwrite,
                        properties_preserve=properties_preserve,
                        properties_overwrite=properties_overwrite,
                        alternate_outputs=require_alternate_output,
                        recursive=recursive)
        output_names = cls.enabled_outputs(self.args)
        output_files = [
            getattr(self.args, f'output_{k}') for k in output_names
        ]
        out = None
        if ((self.output_exists(name=output_names)
             and return_alternate_output is False)):
            if dont_load_existing:
                self.log(f'Output already exists and overwrite '
                         f'not set: {output_files} '
                         f'(output_names={output_names})',
                         cls=cls, force=True)
            else:
                self.log(f'Loading existing output {output_files}',
                         cls=cls, force=True)
                out = cls.read_output(
                    self,
                    require_alternate_output=require_alternate_output
                )
        else:
            # outputs = {k: getattr(self.args, f'output_{k}') for k in
            #            output_names}
            # self.log(f'outputs = {pprint.pformat(outputs)}')
            self.log(f'Generating output {output_files}', cls=cls,
                     force=True)
            out = cls._run(self, **kwargs)
            cls.write_output(self, out)
        self.restore_args()
        if output_name != cls._name:
            self.log(f'Returning alternate value for '
                     f'{output_name}', cls=cls)
            out = self.get_alternate_output(output_name)
        if not dont_reset_alternate_output:
            self._alternate_output.clear()
        return out

    def run(self, **kwargs):
        r"""Run the process associated with this subparser.

        Args:
            **kwargs: Additional keyword arguments are passed to
                run_class.

        Returns:
            object: Generated object.

        """
        kwargs.setdefault('recursive', False)
        return self.run_class(self, **kwargs)

    def add_alternate_output(self, key, value):
        r"""Add an alternate value for return by run_class.

        Args:
            key (str): Name of alternate output.
            value (object): Value of alternate output.

        """
        assert key not in self._alternate_output
        self._alternate_output[key] = value

    def pop_alternate_output(self, key, default=NoDefault,
                             preserve=False):
        r"""Pop an alternate output value from the registry.

        Args:
            key (str): Name of alternate output.
            default (object, optional): Value to return if it is not
                present.
            preserve (bool, optional): If True, don't remove the entry.

        Returns:
            object: Alternate output.

        Raises:
            KeyError: If key is not present and default is not provided.

        """
        if key in self._alternate_output:
            if preserve:
                out = self._alternate_output[key]
            else:
                out = self._alternate_output.pop(key)
        else:
            out = default
        if out is NoDefault:
            raise KeyError(key)
        return out

    def get_alternate_output(self, key, default=NoDefault):
        r"""Get an alternate value for return by run_class.

        Args:
            key (str): Name of alternate output.
            default (object, optional): Value to return if it is not
                present.

        Returns:
            object: Alternate output.

        Raises:
            KeyError: If key is not present and default is not provided.

        """
        return self.pop_alternate_output(key, default=default,
                                         preserve=True)

    @classmethod
    def read_output(cls, self, name=None,
                    require_alternate_output=None):
        r"""Load an output file produced by this task.

        Args:
            self (object): Task instance that is running.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.
            require_alternate_output (list, optional): Set of alternate
                output values that should be read. If not provided,
                the set of required alternate values for the class will
                be used.

        Returns:
            object: Contents of the output file.

        """
        is_base = (name is None)
        if name is None:
            name = cls._name
        outputfile = getattr(self.args, f'output_{name}')
        assert outputfile
        self.log(f'Loading existing output from \"{outputfile}\"',
                 cls=cls)
        out = cls._read_output(self.args, name=name)
        if require_alternate_output is None:
            require_alternate_output = cls._alternate_outputs_write_required
        if is_base and require_alternate_output:
            for k in require_alternate_output:
                koutput = cls.read_output(self, name=k)
                self.add_alternate_output(k, koutput)
        return out

    @classmethod
    def write_output(cls, self, output, name=None):
        r"""Write to an output file.

        Args:
            self (object): Task instance that is running.
            output (object): Output object to write to file.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        """
        is_base = (name is None)
        if name is None:
            name = cls._name
        outputfile = getattr(self.args, f'output_{name}')
        if not outputfile:
            return
        self.log(f'Writing output to \"{outputfile}\"', cls=cls)
        cls._write_output(output, self.args, name=name)
        if is_base:
            for k in cls.enabled_outputs(self.args, no_base=True):
                koutput = self.get_alternate_output(k)
                cls.write_output(self, koutput, name=k)

    @classmethod
    def output_names(cls, only_required=False):
        r"""Get the set of outputs associated with this subparser.

        Args:
            only_required (bool, optional): If True, only include
                required outputs.

        Returns:
            list: Output names.

        """
        out = [cls._name]
        if only_required:
            out += cls._alternate_outputs_write_required
        else:
            out += cls._alternate_outputs_write
        return out

    @classmethod
    def enabled_outputs(cls, args, no_base=False):
        r"""Get the set of outputs enabled by a set of arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            no_base (bool, optional): If True, don't include the default
                output name.

        Returns:
            list: Names of enabled outputs.

        """
        out = []
        if not no_base:
            out += [cls._name]
        out += cls._alternate_outputs_write_required
        out += [
            k for k in cls._alternate_outputs_write_optional
            if getattr(args, f'output_{k}', False)
        ]
        return out

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
        raise NotImplementedError

    @classmethod
    def adjust_args(cls, args):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        for k, v in cls._excluded_arguments_defaults.items():
            if not hasattr(args, k):
                setattr(args, k, v)
        cls.adjust_args_output(args)
        base_overwrite = getattr(args, f'overwrite_{cls._name}')
        for k in cls._alternate_outputs_write:
            if base_overwrite:
                setattr(args, f'overwrite_{k}', True)
            cls.adjust_args_output(args, name=k)

    @classmethod
    def default_filename_applies(cls, args):
        r"""Check if the default file name can be applied by comparing
        the provided arguments against the assumed defaults.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Returns:
            bool: True if the default name applies.

        """
        for k, expected in cls.argument_defaults(use_dest=True).items():
            if k in cls._arguments_suffix_ignore:
                continue
            actual = getattr(args, k)
            match = (actual == expected)
            if isinstance(match, np.ndarray):
                match = np.all(match)
            if not match:
                print(f'Mismatch {k}: {actual} vs. {expected}')
                pdb.set_trace()
                return False
        return True

    @classmethod
    def adjust_args_output(cls, args, name=None):
        r"""Adjust the parsed arguments controling output, generating a
        file name from the other arguments if it is not present.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        """
        if name is None:
            name = cls._name
        output_var = f'output_{name}'
        x = getattr(args, output_var)
        if x is False:
            return
        elif not isinstance(x, str):
            if ((getattr(args, '_in_argument_defaults', False)
                 and (not cls.default_filename_applies(args)))):
                setattr(args, output_var, False)
                return
            directory = cls.output_dir(args, name=name)
            base = cls.output_base(args, name=name)
            suffix = cls.output_suffix(args, name=name)
            if suffix is False:
                setattr(args, output_var, False)
                return
            ext = cls.output_ext(args, name=name)
            x = utils.generate_filename(
                base, ext=ext, suffix=suffix, directory=directory,
            )
            setattr(args, output_var, x)
            # cls.log_class(f'Using generated value for {output_var}: '
            #               f'\"{x}\"')
        overwrite_var = f'overwrite_{name}'
        if args.overwrite_all:
            setattr(args, overwrite_var, True)
        # TODO: Remove all dependencies?
        if getattr(args, overwrite_var) and os.path.isfile(x):
            cls.log_class(f'Removing existing {output_var}: \"{x}\"')
            os.remove(x)

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
        if cls._output_dir is None and cls._name is not None:
            return os.path.join(utils._output_dir, cls._name)
        return cls._output_dir

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
        if cls._name is None:
            raise NotImplementedError
        return cls._name

    @classmethod
    def output_suffix(cls, args, name=None):
        r"""Generate the suffix containing information about parameters
        that should be added to generated output files.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Base name for variable to set. Defaults
                to the task make.

        Returns:
            str: Suffix.

        """
        raise NotImplementedError

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
        return cls._ext

    @classmethod
    def _run(cls, self):
        raise NotImplementedError

    @cached_property
    def figure(self):
        r"""Matplotlib figure."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import matplotlib.units as munits
        converter = mdates.ConciseDateConverter()
        munits.registry[datetime] = converter
        return plt.figure()

    @cached_property
    def axes(self):
        r"""Matplotlib axes."""
        ax = self.figure.add_subplot(111)
        return ax

    @property
    def raw_figure_data(self):
        r"""np.ndarray: Raw pixel data for the current figure."""
        fig = self.figure
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data


def TemporalTaskBase(step_task, step_alias=None):
    r"""Factory for creating a time-step based base class.

    Args:
        step_task (TaskBase): Class for task that will be performed for
            each time step.
        step_alias (str, optional): Alias name for time steps within the
            base class that will be returned.

    Returns:
        type: New base class.

    """

    classname = f'TemporalTaskBase_{step_task._name}'

    def add_step_alias(key):
        if step_alias is None:
            return key
        if isinstance(key, tuple):
            out = list(key)
            for k in key:
                out.append(add_step_alias(k)[-1])
            return tuple(out)
        return tuple([key, key.replace('step', step_alias)])

    class TemporalTask(step_task):
        r"""Base class for performing a task for a set of times."""

        __name__ = classname
        __qualname__ = classname
        _name = None
        _step_task = step_task
        _step_alias = step_alias
        _time_vars = ['start_time', 'stop_time']
        _hour_defaults = {}
        _arguments_suffix_ignore = [
            'start_time', 'stop_time',
        ]
        _alternate_outputs_write_required = []
        _alternate_outputs_write_optional = []
        _arguments = [
            (('--start-time', ), {
                'type': str, 'default': 'sunrise',
                'help': ('Date time (in any ISO 8601 format) that the '
                         'animation should start at. If not provided, the '
                         'time of sunrise for the selected \"location\" (or '
                         '\"latitude\"/\"longitude\") & \"doy\" will be '
                         'used.'),
            }),
            (('--stop-time', ), {
                'type': str, 'default': 'sunset',
                'help': ('Date time (in any ISO 8601 format) that the '
                         'animation should stop at. If not provided, the '
                         'time of sunset for the selected \"location\" (or '
                         '\"latitude\"/\"longitude\") & \"doy\" will be '
                         'used.'),
            }),
            (add_step_alias(('--step-count', )), {
                'type': int,
                'help': ('The number of time steps that should be taken '
                         'between the start and end time. If not provided, '
                         'the number of time steps will be determined from '
                         '\"step_interval\"'),
            }),
            (add_step_alias(('--step-interval', )), {
                'type': parse_quantity, 'units': 'hours',
                'help': ('The interval (in hours) that should be used '
                         'between time steps. If not provided, '
                         '\"step_count\" will be used to calculate the '
                         'step interval. If \"step_count\" is not '
                         'provided, a step interval of 1 hour will be '
                         'used.'),
            }),
            (add_step_alias((
                f'--output-{step_task._name}', '--output-steps'
            )), {
                'action': 'store_true',
                'help': 'Output the step data to disk',
            }),
            (add_step_alias((
                f'--overwrite-{step_task._name}', '--overwrite-steps'
            )), {
                'action': 'store_true',
                'help': 'Regenerate step data that already exist',
            }),
        ]
        _excluded_arguments = [
            '--time',
        ]

        @classmethod
        def adjust_args(cls, args, **kwargs):
            r"""Adjust the parsed arguments including setting defaults that
            depend on other provided arguments.

            Args:
                args (argparse.Namespace): Parsed arguments.

            """
            super(TemporalTask, cls).adjust_args(args, **kwargs)
            duration = args.stop_time - args.start_time
            duration = duration.total_seconds() / 3600
            if not args.step_count:
                if not args.step_interval:
                    args.step_interval = 1.0
                args.step_count = int(duration / args.step_interval)
            elif not args.step_interval:
                args.step_interval = duration / args.step_count

        @classmethod
        def adjust_args_time(cls, args, timevar=None):
            r"""Adjust the time related variables in a set of parsed
            arguments.

            Args:
                args (argparse.Namespace): Parsed arguments.
                timevar (str, optional): Time variable to adjust. If not
                    provided, all of the time variables associated with this
                    subparser will be adjusted.

            """
            super(TemporalTask, cls).adjust_args_time(args, timevar=timevar)
            if timevar is not None:
                return
            if args.stop_time == args.start_time:
                args.stop_time = args.stop_time.replace(hour=0, minute=0,
                                                        microsecond=0)
                cls.adjust_args_time(args, timevar='stop_time')
            start_time_str = getattr(args, 'start_time_str', None)
            stop_time_str = getattr(args, 'stop_time_str', None)
            if start_time_str and stop_time_str:
                start_parts = start_time_str.rsplit('-', 1)
                stop_parts = stop_time_str.rsplit('-', 1)
                if start_parts[0] == stop_parts[0]:
                    args.stop_time_str = stop_parts[-1]
            if not hasattr(args, 'time'):
                args.time = None
                args.time_str = None

        @cached_factory_property(classname)
        def times(self):
            r"""list: Set of times for steps in the task."""
            dt = timedelta(hours=self.args.step_interval)
            time = self.args.start_time
            out = []
            for i in range(self.args.step_count):
                out.append(time)
                time += dt
            if time < self.args.stop_time:
                out.append(self.args.stop_time)
            return out

        @classmethod
        def _run_step(cls, self, time, **kwargs):
            if cls._step_task is None:
                raise NotImplementedError
            step_output = f'output_{cls._step_task._name}'
            kwargs.setdefault('args_overwrite', {})
            kwargs.setdefault('args_preserve', [])
            kwargs['args_overwrite']['time'] = time.isoformat()
            kwargs['args_overwrite'].setdefault(step_output, True)
            kwargs['args_preserve'].append(step_output)
            return cls._step_task.run_class(self, **kwargs)

        @classmethod
        def _run(cls, self):
            r"""Run the process associated with this subparser."""
            result = []
            if ((cls._output_dir is not None
                 and not os.path.isdir(cls._output_dir))):
                os.mkdir(cls._output_dir)
            for time in self.times:
                result.append(cls._run_step(self, time))
            return result

    return TemporalTask


#################################################################
# CLI
#################################################################

def main(**kwargs):
    r"""Parse arguments provided via the command line or keyword
    arguments

    Args:
        **kwargs: If any keyword args are passed, they are parsed
            instead of the command line arguments.

    Returns:
        argparse.Namespace, dict: Argument namespace and keyword
            keyword arguments that were not parsed.

    """
    parser = InstrumentedParser("Generate a 3D maize model")
    for v in get_class_registry().values('task'):
        v.add_arguments(parser)
    if kwargs:
        arglist = [kwargs.get('task', 'generate')]
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()
    for k in list(kwargs.keys()):
        if hasattr(args, k):
            setattr(args, k, kwargs.pop(k))
    parser.run_subparser('task', args)
    return args, kwargs


if __name__ == "__main__":
    args, _ = main()
