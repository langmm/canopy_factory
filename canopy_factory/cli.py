import os
import gc
import sys
import pdb
import copy
import pprint
import argparse
import numpy as np
import inspect
import pytz
import glob
import itertools
from collections import OrderedDict
from datetime import datetime
from canopy_factory import utils
from canopy_factory.utils import (
    cfg, rapidjson, units, parse_quantity, format_list_for_help,
    get_class_registry, NoDefault, RegisteredClassBase,
    cached_property,
)


class SetBase(object):
    r"""Simple wrapper for a set of classes.

    Args:
        members (list): Set of instances of the same class.
        parent (object, optional): Instance containing this group.

    """

    _set_attributes = ['parent', 'members']

    def __init__(self, parent, members):
        self.parent = parent
        self.members = members

    @classmethod
    def _is_set_attribute(cls, k):
        return (k in cls._set_attributes or hasattr(cls, k))

    def __getattr__(self, k):
        if type(self)._is_set_attribute(k):
            return super(SetBase, self).__getattr__(k)
        if not self.members:
            raise AttributeError(f'{self}: {k}')
        vals = [getattr(x, k) for x in self.members]
        out = vals[0]
        if not all([x == out] for x in vals):
            raise AttributeError(f'{self}: {k} (values differ {vals})')
        return out

    def __setattr__(self, k, v):
        if type(self)._is_set_attribute(k):
            return super(SetBase, self).__setattr__(k, v)
        if not self.members:
            raise AttributeError(f'{self}: {k}')
        for x in self.members:
            setattr(x, k, v)


class InstrumentedParserSet(SetBase):
    r"""Simple wrapper for a set of parsers.

    Args:
        members (list): Set of parsers.
        parent (ArgumentParser, optional): Parser containing this group.

    """

    def add_subparsers(self, group, **kwargs):
        r"""Add a subparsers group to this parser.

        Args:
            group (str): Name of the subparser group. Used to set the
                default 'title' & 'dest' subparser properties.
            **kwargs: Additional keyword arguments are passed to the base
                class's method.

        Returns:
            InstrumentedParserSet: Wrapped subparsers group.

        """
        return InstrumentedSubparserSet(
            self,
            [x.add_subparsers(group, **kwargs) for x in self.members]
        )

    def add_argument(self, *args, **kwargs):
        r"""Add an argument to the parser.

        Args:
            *args, **kwargs: All arguments are passed to the add_argument
                method of the subparsers (if there are any) or the parent
                class (if there are not any subparsers).

        """
        return ActionSet(
            self, [
                x.add_argument(*args, **kwargs) for x in self.members
            ]
        )


class InstrumentedSubparserSet(InstrumentedParserSet):
    r"""Simple wrapper for a set of subparsers."""

    @property
    def choices(self):
        r"""dict: Set of subparsers in the group."""
        return {
            k: InstrumentedParserSet(
                self, [x.choices[k] for x in self.members])
            for k in self.members[0].choices.keys()
        }

    def has_parser(self, name):
        r"""Check if there is a parser for a given name within this
        subparser group.

        Args:
            name (str): ID string for the parser.

        Returns:
            bool: True if the name parser exists, False otherwise.

        """
        return all(x.has_parser(name) for x in self.members)

    def add_parser(self, name, *args, **kwargs):
        r"""Add a parser to the subparser group.

        Args:
            name (str): ID string for the parser.
            *args, **kwargs: Additional arguments are passed to the
                add_parser method for the underlying subparser group.

        Returns:
            ArgumentParser: New subparser.

        """
        return InstrumentedParserSet(
            self,
            [x.add_parser(name, *args, **kwargs) for x in self.members]
        )

    def get_parser(self, name, *args, **kwargs):
        r"""Get a parser for an option in the subparser group.

        Args:
            name (str): ID string for the parser.
            default (object, optional): Default to return if the
                subparser instance does not exist.
            add_missing (bool, dict, optional): If True or dictionary,
                add the subparser if it does not exist. If a dictionary
                is provided, it will be passed as keyword arguments to
                the add_parser method.
            *args, **kwargs: Additional arguments are passed to the
                get_parser method for the underlying subparser group.

        Returns:
            ArgumentParser: Subparser.

        """
        return InstrumentedParserSet(
            self,
            [x.get_parser(name, *args, **kwargs) for x in self.members]
        )


class ActionSet(SetBase):
    r"""Simple wrapper for a set of actions."""
    pass


class SubParsersAction(argparse._SubParsersAction):

    def __init__(self, *args, default=None, choices=None, func=None,
                 positional_index=0, subparser_name=None, **kwargs):
        super(SubParsersAction, self).__init__(*args, **kwargs)
        self.subparser_default = default
        self.func = func
        self._choice_names = []
        self._positional_index = positional_index
        self._subparser_name = subparser_name
        if choices:
            for k in choices:
                self.add_parser(k)

    def has_parser(self, name):
        r"""Check if there is a parser for a given name within this
        subparser group.

        Args:
            name (str): ID string for the parser.

        Returns:
            bool: True if the name parser exists, False otherwise.

        """
        return (name in self._name_parser_map)

    def add_parser(self, name, *args, **kwargs):
        r"""Add a parser to the subparser group.

        Args:
            name (str): ID string for the parser.
            *args, **kwargs: Additional arguments are passed to the
                add_parser method for the underlying subparser group.

        Returns:
            ArgumentParser: New subparser.

        """
        kwargs['positional_index'] = self._positional_index + 1
        self._choice_names.append(name)
        out = super(SubParsersAction, self).add_parser(
            name, *args, **kwargs)
        out._subparser_name = name
        out._subparser_group = self
        return out

    def get_parser(self, name, default=NoDefault, add_missing=False):
        r"""Get a parser for an option in the subparser group.

        Args:
            name (str): ID string for the parser.
            default (object, optional): Default to return if the
                subparser instance does not exist.
            add_missing (bool, dict, optional): If True or dictionary,
                add the subparser if it does not exist. If a dictionary
                is provided, it will be passed as keyword arguments to
                the add_parser method.

        Returns:
            ArgumentParser: Subparser.

        """
        if not self.has_parser(name):
            if add_missing is True:
                add_missing = {}
            if isinstance(add_missing, dict):
                self.add_parser(name, **add_missing)
            elif default != NoDefault:
                return default
        return self._name_parser_map[name]

    def add_subparsers(self, group, **kwargs):
        r"""Add a subparsers group to each parser in this subparser
        group.

        Args:
            group (str): Name of the subparser group. Used to set the
                default 'title' & 'dest' subparser properties.
            **kwargs: Additional keyword arguments are passed to the base
                class's method.

        Returns:
            InstrumentedParserSet: Set of subparsers for each parser.

        """
        return InstrumentedSubparserSet(
            self, [
                self._name_parser_map[k].add_subparsers(group, **kwargs)
                for k in self._choice_names
            ]
        )

    def add_argument(self, *args, **kwargs):
        r"""Add an argument to the parser.

        Args:
            *args, **kwargs: All arguments are passed to the add_argument
                method of the subparsers (if there are any) or the parent
                class (if there are not any subparsers).

        """
        return ActionSet(
            self, [
                self._name_parser_map[k].add_argument(*args, **kwargs)
                for k in self._choice_names
            ]
        )


class InstrumentedParser(argparse.ArgumentParser):
    r"""Class for parsing arguments allowing arguments to be
    added to multiple subparsers."""

    def __init__(self, *args, parent=None, positional_index=0, **kwargs):
        self._subparsers_action = None
        self._subparser_name = None
        self._subparser_group = None
        self._child_subparsers = {}
        self._positional_index = positional_index
        self._quantity_units = {}
        super(InstrumentedParser, self).__init__(*args, **kwargs)
        self.register('action', 'parsers', SubParsersAction)
        self.add_argument(
            '--hide-irrelevant', default=-1,
            dest='hide_irrelevant', type=int, const=0, nargs='?',
            help=('Hide arguments that are diabled by absent or '
                  'invalid arguments already specified. This can '
                  'be useful for getting more information about '
                  'available command line options'),
        )

    def local_subparsers(self, subparsers=None):
        r"""Get the subparser instances for the subparser group directly
        managed by this parser (if there is one).

        Args:
            subparsers (list, optional): Names of subparsers that should
                be included in the returned list. If not provided, all
                subparsers in the group will be returned.

        Returns:
            list: Subparser instances.

        """
        if self._subparsers_action is None:
            return []
        if isinstance(subparsers, dict) and self._subparsers_action:
            subparsers = subparsers.get(self._subparsers_action.dest, None)
        if subparsers is None:
            return list(self._subparsers_action.choices.values())
        return [v for k, v in self._subparsers_action.choices.items()
                if k in subparsers]

    def subparsers(self, group=None, subparsers=None, yield_roots=False):
        r"""Iterate over all subparser instances within this parser.

        Args:
            group (str, optional): Group that subparsers should be
                returned for. If not provided, all leaf level subparsers
                will be yielded.
            subparsers (dict, optional): Maps between the names of
                subparser groups and the names of subparsers within
                those groups that should be included.
            yield_roots (bool, optional): If True, yield the highest
                level parser that matches the group.

        Yields:
            InstrumentedParser: Subparsers.

        """
        if group is None and (self._subparsers_action is None
                              or yield_roots):
            yield self
            return
        if subparsers is None:
            subparsers = {}
        if ((self._subparsers_action is not None
             and self._subparsers_action.dest == group)):
            if yield_roots:
                for x in self.local_subparsers(subparsers=subparsers):
                    yield x
                return
            group = None  # Allow children to be yielded
        for x in self.local_subparsers(subparsers=subparsers):
            if yield_roots and group is None:
                yield x
            else:
                for xx in x.subparsers(group=group, subparsers=subparsers,
                                       yield_roots=yield_roots):
                    yield xx

    def _action_irrelevant(self, action, args):
        if action.help == argparse.SUPPRESS:
            return False
        if not getattr(action, 'dependencies', None):
            return False
        out = False
        for i, x in enumerate(action.dependencies[::-1]):
            if not x:
                continue
            for k, v in x.items():
                add_to_parent = (i == 0)
                argv = getattr(args, k, None)
                if (((v is True and argv is None)
                     or (v is False and argv is not None)
                     or (isinstance(v, (list, tuple)) and argv not in v))):
                    if args.hide_irrelevant > 0:
                        if i >= args.hide_irrelevant:
                            out = True
                        add_to_parent = (i <= args.hide_irrelevant)
                    else:
                        out = True
                if not add_to_parent:
                    continue
                if v is True and argv is None:
                    self.find_argument(k).enables_dependencies.append(
                        action.dest)
                elif v is False and argv is not None:
                    self.find_argument(k).disables_dependencies.append(
                        action.dest)
                elif isinstance(v, (list, tuple)) and argv not in v:
                    parent_action = self.find_argument(k)
                    for kdep in v:
                        parent_action.optional_dependencies.setdefault(
                            kdep, [])
                        parent_action.optional_dependencies[kdep].append(
                            action.dest)
        return out

    def prune_irrelevant_args(self, args):
        r"""Remove dependent arguments from the parser that are not
        valid for the provided arguments.

        Args:
            args (argparse.Namespace): Arguments parsed so far.

        Returns:
            bool: True if arguments were pruned, False otherwise.

        """
        if args.hide_irrelevant == -1:
            return False
        changes = False
        for action in self._actions:
            if self._action_irrelevant(action, args):
                action.help = argparse.SUPPRESS
                action.default = argparse.SUPPRESS
                changes = True
        for x in self.local_subparsers():
            if x.prune_irrelevant_args(args):
                changes = True
        if changes:
            self._add_irrelevant_args_to_help(args)
        return changes

    def _add_irrelevant_args_to_help(self, args):
        for action in self._actions:
            if action.help == argparse.SUPPRESS:
                continue
            new_help = []
            if getattr(action, 'enables_dependencies', None):
                vals = format_list_for_help(
                    sorted(list(set(action.enables_dependencies))))
                new_help.append(f'Enables {vals}.')
                action.enables_dependencies = []
            if getattr(action, 'disables_dependencies', None):
                vals = format_list_for_help(
                    sorted(list(set(action.disables_dependencies))))
                new_help.append(f'Disables {vals}.')
                action.disables_dependencies = []
            if getattr(action, 'optional_dependencies', None):
                values = [
                    f'\'{k}\' enables {format_list_for_help(v)}.'
                    for k, v in action.optional_dependencies.items()
                ]
                new_help.append('\n' + '\n'.join(values))
                action.optional_dependencies = {}
            if new_help:
                if not action.help.endswith('.'):
                    action.help += '.'
                action.help += ' ' + ' '.join(new_help)
        for x in self.local_subparsers():
            x._add_irrelevant_args_to_help(args)

    def add_subparser_defaults(self, args):
        if self._subparsers_action is None:
            return None
        value = getattr(args, self._subparsers_action.dest, None)
        if value is None:
            if self._subparsers_action.subparser_default is None:
                return None
            return (
                self._subparsers_action._positional_index,
                self._subparsers_action.subparser_default
            )
        return self._subparsers_action.choices[
            value].add_subparser_defaults(args)

    def _parse_args_partial(self, args, for_help=False):
        r"""Parse arguments, allowing for modifications to either the
        input arguments or argument parser based on the arguments that are
        present.

        Args:
            args (list): Arguments to parse.
            for_help (bool, optional): If True, the arguments are being
                parsed to generate help and defaults should not be set for
                subparsers.

        Returns:
            bool: True if either the arguments or argument parser was
                modified.

        """
        assert isinstance(args, list)
        changes = False
        args0, _ = super(InstrumentedParser, self).parse_known_args(
            args=args)
        if self.prune_irrelevant_args(args0):
            changes = True
        if for_help:
            pass
        else:
            args_supp = self.add_subparser_defaults(args0)
            if args_supp:
                changes = True
                args.insert(*args_supp)
        return changes

    def parse_known_args(self, args=None, namespace=None, **kwargs):
        r"""Parse known arguments.

        Args:
            args (list, optional): Arguments to parse. Defaults to
                sys.argv if not provided.
            namespace (argparse.Namespace, optional): Existing namespace
                that arguments should be added to.
            **kwargs: Additional keyword arguments are passed to the
                parent class's method.

        Returns:
            tuple(argparse.Namespace, list): Parsed and unparsed
                arguments.

        """
        if self._subparser_group is None:
            if args is None:
                args = sys.argv[1:]
            help_flags = ['-h', '--help']
            for_help = False
            for x in help_flags:
                if x in args:
                    for_help = x
                    args.remove(x)
                    break
            while self._parse_args_partial(args, for_help=for_help):
                continue
            if for_help:
                args.append(for_help)
        out, argv = super(InstrumentedParser, self).parse_known_args(
            args=args, namespace=namespace, **kwargs)
        out = self.add_units(out)
        return out, argv

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
        for x in self.local_subparsers():
            x.add_units(args)
        return args

    def has_subparser(self, group, name):
        r"""Check if there is a named subparser within a subparsers group.

        Args:
            group (str): Name of the subparsers group to check for.
            name (str): Name of the parser within the subparsers group to
                check for.

        Returns:
            bool: True if the named parser within the subparsers group
                exists.

        """
        return (self.has_subparsers(group)
                and self.get_subparsers(group).has_parser(name))

    def has_subparsers(self, group):
        r"""Check if there is a subparsers group associated with the
        specified name.

        Args:
            group (str): Name of the subparsers group to check for.

        Returns:
            bool: True if the named subparsers group exists.

        """
        return (group in self._child_subparsers)

    def get_subparser(self, group, name, default=NoDefault,
                      add_missing=False, add_missing_group=None):
        r"""Get a single subparser from a subparser group.

        Args:
            group (str): Name of the subparser group that the parser
                belongs to.
            name (str): Name of the subparser.
            default (object, optional): Default to return if the
                subparser instance does not exist.
            add_missing (bool, dict, optional): If True or dictionary,
                add the subparser if it does not exist. If a dictionary
                is provided, it will be passed as keyword arguments to
                the add_subparser method.
            add_missing_group (bool, dict, optional): If True or
                dictionary, the subparser group will be added if it is
                missing. If a dictionary is provided it will be passed as
                keyword arguments to the add_subparsers method. If not
                provided, add_missing_group will be set to True only if
                add_missing is True or a dictionary.

        Returns:
            InstrumentedParser: Parser.

        Raises:
            KeyError: If default not provided and the subparser object
                does not exist.

        """
        if add_missing_group is None:
            add_missing_group = (
                add_missing or isinstance(add_missing, dict))
        subparsers = self.get_subparsers(group, default=default,
                                         add_missing=add_missing_group)
        if not self.has_subparsers(group):
            return subparsers  # Will be default
        return subparsers.get_parser(name, default=default,
                                     add_missing=add_missing)

    def get_subparsers(self, group, default=NoDefault, add_missing=False):
        r"""Get the subparsers object with the specified group name.

        Args:
            group (str): Name of the subparsers instance to retrieve.
            default (object, optional): Default to return if the
                subparsers instance does not exist.
            add_missing (bool, dict, optional): If True or dictionary, add
                the subparser group if it does not exist. If a dictionary
                is provided, it will be passed as keyword arguments to
                the add_subparsers method.

        Returns:
            InstrumentedSubparsers: Subparsers instance.

        Raises:
            KeyError: If default not provided and the subparsers object
                does not exist.

        """
        out = default
        if add_missing is True:
            add_missing = {}
        if ((isinstance(add_missing, dict)
             and group not in self._child_subparsers)):
            # TODO: Use name of first subparser for default?
            self.add_subparsers(group, **add_missing)
        if group in self._child_subparsers:
            out = self._child_subparsers[group]
        if out is NoDefault:
            raise KeyError(group)
        return out

    def add_subparser(self, group, name, add_missing_group=True,
                      **kwargs):
        r"""Add a subparser to a subparsers group.

        Args:
            group (str): Name of the subparser group that the parser
                belongs to.
            name (str): Name of the subparser.
            add_missing_group (bool, dict, optional): If True or
                dictionary, the subparser group will be added if it is
                missing. If a dictionary is provided it will be passed as
                keyword arguments to the add_subparsers method.
            **kwargs: Additional keyword argumetns are passed to the
                add_parser method of the wrapped subparsers group.

        Returns:
            InstrumentedParser: Parser.

        """
        assert not self.has_subparser(group, name)
        subparsers = self.get_subparsers(
            group, add_missing=add_missing_group)
        subparsers.add_parser(name, **kwargs)
        return self.get_subparser(group, name)

    def add_subparsers(self, group, **kwargs):
        r"""Add a subparsers group to this parser.

        Args:
            group (str): Name of the subparser group. Used to set the
                default 'title' & 'dest' subparser properties.
            **kwargs: Additional keyword arguments are passed to the base
                class's method.

        Returns:
            InstrumentedSubparsers: Wrapped subparsers group.

        """
        assert group not in self._child_subparsers
        kwargs.setdefault('dest', group)
        kwargs.setdefault('title', group)
        if self._subparsers_action is not None:
            out = self._subparsers_action.add_subparsers(group, **kwargs)
        else:
            positional_index = (
                self._positional_index
                + len(self._get_positional_actions())
            )
            subparser_name = self._subparser_name
            out = super(InstrumentedParser, self).add_subparsers(
                positional_index=positional_index,
                subparser_name=subparser_name, **kwargs)
            self._subparsers_action = out
        self._child_subparsers[group] = out
        return out

    def subparser_class(self, group, args):
        r"""Get the subparser class selected by a subparser group.

        Args:
            group (str): Name of the subparser group to get the class for.
            args (argparse.Namespace): Parsed arguments.

        Returns:
            type: Subparser class.

        """
        subparser = self.get_subparsers(group)
        subparser_name = getattr(args, group)
        if subparser_name is None:
            subparser_name = subparser.default
            setattr(args, group, subparser_name)
        assert subparser.func is None
        return get_class_registry().get(group, subparser_name)

    def construct_subparser(self, group, args, **kwargs):
        r"""Construct the subparser selected by a subparser group.

        Args:
            group (str): Name of the subparser group to construct.
            args (argparse.Namespace): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to the
                class constructor.

        Returns:
            SubparserBase: Subparser instance.

        """
        cls = self.subparser_class(group, args)
        return cls(args, **kwargs)

    def run_subparser(self, group, args, return_func=False):
        r"""Run the subparser selected by a subparser group.

        Args:
            group (str): Name of the subparser group to run.
            args (argparse.Namespace): Parsed arguments.
            return_func (bool, optional): If True, return the subparser
                run function without calling it.

        """
        subparser = self.get_subparsers(group)
        subparser_name = getattr(args, group)
        if subparser_name is None:
            subparser_name = subparser.default
        func = subparser.func
        if func is None:
            func = get_class_registry().get(group, subparser_name)
        if return_func:
            return func
        print(f"RUNNING {subparser_name}")
        return func(args)

    def find_argument(self, name, default=NoDefault):
        r"""Find the action that will handle an argument.

        Args:
            name (str): Argument name or flag.
            default (object, optional): Object that should be returned
                if the arguments cannot be located.

        Returns:
            Action: Argument action.

        Raises:
            KeyError: If the argument cannot be located and a default is
                not provided.

        """
        for action in self._actions:
            if ((name in action.option_strings or name == action.dest
                 or name.lstrip('-').replace('-', '_') == action.dest)):
                return action
        if default is NoDefault:
            raise KeyError(name)
        return default

    def has_argument(self, name):
        r"""Check to see if the parser has an argument.

        Args:
            name (str): Argument name or flag.

        Returns:
            bool: True if the parser has an argument matching the name,
                False otherwise.

        """
        try:
            self.find_argument(name)
            return True
        except KeyError:
            return False

    def remove_argument(self, name):
        r"""Remove an argument from the parser.

        Args:
            name (str): Name of the argument to remove.

        """
        if isinstance(name, str):
            action = self.find_argument(name)
        else:
            action = name
        action.container._remove_action(action)
        for k in action.option_strings:
            action.container._option_string_actions.pop(k)

    def add_argument(self, *args, **kwargs):
        r"""Add an argument to the parser.

        Args:
            *args, **kwargs: All arguments are passed to the add_argument
                method of the subparsers (if there are any) or the parent
                class (if there are not any subparsers).

        """
        kwargs.setdefault('dest', args[0].lstrip('-').replace('-', '_'))
        kwargs = copy.deepcopy(kwargs)
        dependencies = kwargs.pop('dependencies', None)
        is_subparser = kwargs.pop("is_subparser", False)
        subparsers = kwargs.pop('subparsers', None)
        subparser_options = kwargs.pop('subparser_options', {})
        subparser_specific_dest = kwargs.pop(
            'subparser_specific_dest', False)
        assert not subparsers
        assert not subparser_options
        if self._subparsers_action is not None:
            x = self._subparsers_action
        else:
            x = self
        iargs = copy.deepcopy(args)
        ikwargs = copy.deepcopy(kwargs)
        ikwargs = dict(kwargs, **subparser_options.get(x, {}))
        if subparser_specific_dest:
            ikwargs['dest'] += f'_{x._subparser_name}'
            iargs = tuple(
                list(iargs) + [args[0] + f'-{x._subparser_name}'])
        if 'units' in ikwargs:
            self._quantity_units[ikwargs['dest']] = ikwargs.pop('units')
        if is_subparser:
            group = iargs[0].replace('-', '_').lstrip('_')
            action = x.add_subparsers(group, **ikwargs)
        elif isinstance(x, InstrumentedParser):
            action = super(InstrumentedParser, x).add_argument(
                *iargs, **ikwargs)
        else:
            action = x.add_argument(*iargs, **ikwargs)
        action.dependencies = dependencies
        action.enables_dependencies = []
        action.disables_dependencies = []
        action.optional_dependencies = {}
        return action


class CompositeArgument(RegisteredClassBase):
    r"""Container for parsing related arguments.

    Args:
        name (str): Base argument name.
        args (argparse.Namespace): Parsed arguments.
        name_base (str): Name of the base variable that should be used
            to set defaults.
        defaults (dict, optional): Defaults that should be used for
            time arguments before the class _defauls.
        ignore (list, optional): Arguments that should be ignored.
        optional (bool, optional): If True, defaults from the class will
            not be set when the argument is not defined.
        description (str, optional): Argument description.
        args_overwrite (dict, optional): Arguments to overwrite.

    Class Attributes:
        _name (str): Name of the base variables that should be
            used to determine the prefix/suffix and where the instance
            should be stored on the parsed arguments.
        _defaults (dict): Argument defaults.
        _arguments (list): Arguments that should be prefixed.
        _arguments_universal (list): Arguments that should not be
            prefixed/suffixed.

    """

    _registry_key = 'argument'
    _name = None
    _name_as_suffix = False
    _name_as_prefix = False
    _defaults = {}
    _arguments = []
    _arguments_universal = []
    _attributes_kwargs = [
        'optional', 'description',
    ]
    _attributes_copy = []

    def __init__(self, name, args, name_base=None, defaults=None,
                 ignore=None, optional=False, description=None,
                 args_overwrite=None):
        self.name = name
        if name_base == name:
            name_base = None
        if defaults is None:
            defaults = {}
        if ignore is None:
            ignore = []
        self._defaults_set = []
        self.name_base = name_base
        self.prefix = self.get_prefix(self.name)
        self.suffix = self.get_suffix(self.name)
        self.base = None
        self.ignore = ignore
        self.optional = optional
        self.description = description
        self.defaults = {
            k: defaults[k] for k in self.argument_names()
            if k in defaults
        }
        self.args = {}
        self.reset(args, args_overwrite=args_overwrite)

    def reset(self, args, args_overwrite=None):
        r"""Reinitialize the arguments used by this instance.

        Args:
            args (argparse.Namespace): Parsed arguments.
            args_overwrite (dict, optional): Arguments to overwrite.

        """
        if args_overwrite is None:
            args_overwrite = {}
        # in_init = bool(self.args)
        self.args.clear()
        self.clear_cached_properties()
        self._defaults_set.clear()
        if self.name_base is not None:
            self.base = self.from_args(
                self.name_base, args, name_base=self.name_base,
                args_overwrite=args_overwrite,
            )
        args_composite = []
        for k in self.argument_names():
            kbook = self.prefix + k + self.suffix
            self.args[k] = (
                None if k in self.ignore
                else args_overwrite.get(kbook,
                                        getattr(args, kbook, None))
            )
        for k in args_composite:
            while isinstance(self.args[k], CompositeArgument):
                v = self.args[k]
                assert k == v._name
                for kk in v.argument_names():
                    if ((kk == k or kk in (self.ignore + v.ignore)
                         or v.args.get(kk, None) is None)):
                        continue
                    assert v.args[kk] == self.args[kk]
                self.args[k] = v.args[k]
        for k in self.ignore:
            self.args[k] = None

    @classmethod
    def from_other(cls, other, **kwargs):
        r"""Create an argument instance from an existing instance of any
        CompositeArgument subclass.

        Args:
            other (CompositeArgument): Instance to copy.
            **kwargs: Additional keyword arguments are passed to the
                class constructor.

        Returns:
            CompositeArgument: Copy as this class.

        """
        argument_names = cls.argument_names()
        name = other.prefix + cls._name + other.suffix
        if other.name_base and 'name_base' not in kwargs:
            kwargs['name_base'] = (
                other.get_prefix(other.name_base) + cls._name
                + other.get_suffix(other.name_base)
            )
        if other.defaults and 'defaults' not in kwargs:
            kwargs['defaults'] = {
                k: v for k, v in other.defaults.items()
                if k in argument_names
            }
        if other.ignore and 'ignore' not in kwargs:
            kwargs['ignore'] = [
                k for k in other.ignore
                if k in argument_names
            ]
        for k in cls._attributes_kwargs:
            if ((k not in other._attributes_kwargs
                 or k in kwargs)):
                continue
            kwargs[k] = getattr(other, k)
        kws = {}
        for k, v in other.args.items():
            if k not in argument_names:
                continue
            kws[other.prefix + k + other.suffix] = v
        out = cls.from_kwargs(name, kws, **kwargs)
        for k in cls._attributes_copy:
            if hasattr(other, k):
                setattr(out, k, getattr(other, k))
        return out

    @classmethod
    def from_kwargs(cls, name, kws, **kwargs):
        r"""Create an argument instance by pulling arguments from the
        provided keyword arguments.

        Args:
            name (str): Base argument name.
            kws (dict): Keywords to add to a new argument namespace.
            **kwargs: Additional keyword arguments are passed to from_args

        Returns:
            CompositeArgument: New instance.

        """
        if name is None:
            name = cls._name
        args = argparse.Namespace()
        arguments = cls.argument_names(name)
        for k, v in kws.items():
            if k not in arguments:
                continue
            setattr(args, k, v)
        return cls.from_args(name, args, **kwargs)

    @classmethod
    def from_args(cls, name, args, overwrite=False, dont_update=False,
                  **kwargs):
        r"""Create an instance on the provided arguments, first checking
        if one already exists.

        Args:
            name (str): Base argument name.
            args (argparse.Namespace): Parsed arguments.
            overwrite (bool, optional): If True, overwrite the existing
                instance.
            dont_update (bool, optional): If True, don't set the argument
                attributes.
            **kwargs: Additional keyword arguments are passed to the
                class constructor.

        Returns:
            CompositeArgument: New or existing instance.

        """
        if overwrite:
            if dont_update:
                args = copy.deepcopy(args)
            cls.reset_args(name, args)
        inst = getattr(args, name, None)
        if not isinstance(inst, cls):
            inst = cls(name, args, **kwargs)
        if not dont_update:
            inst.update_args(args)
        return inst

    @classmethod
    def reset_args(cls, name, args, value=NoDefault):
        r"""Reset the arguments for a variable.

        Args:
            name (str): Variable name.
            args (argparse.Namespace): Parsed arguments.
            value (object, optional): Value that the parsed variable
                should be reset to.

        """
        if value is NoDefault:
            if isinstance(getattr(args, name, None), CompositeArgument):
                value = getattr(args, name).args.get(name, None)
            else:
                value = getattr(args, name)
        setattr(args, name, value)

    def update_args(self, args, name=None):
        r"""Update a namespace with the parsed arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Name that should be used for the set
                variables.

        """
        if name is None:
            name = self.name
        setattr(args, name, self)

    @classmethod
    def get_prefix(cls, name):
        r"""Get the prefix used for a variable.

        Args:
            name (str): Variable name.

        Returns:
            str: Variable prefix.

        """
        if name == cls._name:
            out = ''
        else:
            out = name.rsplit(cls._name, 1)[0]
        if cls._name_as_prefix:
            out += cls._name
        return out

    @classmethod
    def get_suffix(cls, name):
        r"""Get the suffix used for a variable.

        Args:
            name (str): Variable name.

        Returns:
            str: Variable suffix.

        """
        if name == cls._name:
            out = ''
        else:
            out = name.split(cls._name, 1)[1]
        if cls._name_as_suffix:
            out = cls._name + out
        return out

    @classmethod
    def argument_names(cls, name=None, prefix=None, suffix=None,
                       no_universal=False):
        r"""Get the argument names used to specify a variable.

        Args:
            name (str, optional): Base name for argument to determine
                prefix & suffix from.
            prefix (str, optional): Prefix that should be added to
                arguments.
            suffix (str, optional): Suffix that should be added to
                arguments.
            no_universal (bool, optional) If True, don't include
                universal arguments.

        Returns:
            list: Argument names.

        """
        if prefix is None or suffix is None:
            assert prefix is None and suffix is None
            if name is None:
                name = cls._name
            prefix = cls.get_prefix(name)
            suffix = cls.get_suffix(name)
        out = [
            prefix + SubparserBase.arg2dest(args, kwargs) + suffix
            for args, kwargs in cls._arguments
        ]
        if no_universal:
            return out
        out += [
            SubparserBase.arg2dest(args, kwargs)
            for args, kwargs in cls._arguments_universal
        ]
        return out

    @classmethod
    def argument_dict(cls, prefix='', suffix='', no_universal=False,
                      description='', ignore=None,
                      use_flags=False, **kwargs):
        r"""Get a dictionary of argument data.

        Args:
            prefix (str, optional): Prefix that should be added to
                arguments.
            suffix (str, optional): Suffix that should be added to
                arguments.
            no_universal (bool, optional) If True, don't include
                universal arguments.
            description (str, optional): Description string that should
                be used to complete the argument help strings.
            ignore (list, optional): Arguments to ignore.
            use_flags (bool, optional): If False, use the argument
                destinations. If True, use the argument flags.
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            dict: Positional and keyword arguments for creating the
                arguments.

        """
        if ignore is None:
            ignore = []
        out = OrderedDict()
        prefix_arg = prefix.replace('_', '-')
        prefix_dst = prefix.replace('-', '_')
        suffix_arg = suffix.replace('_', '-')
        suffix_dst = suffix.replace('-', '_')
        for iargs, ikwargs in cls._arguments:
            dest0 = SubparserBase.arg2dest(iargs, ikwargs)
            if dest0 in ignore:
                continue
            if prefix_arg or suffix_arg:
                iargs = tuple([
                    f'--{prefix_arg}' + k.split('--', 1)[-1] + suffix_arg
                    for k in iargs if k.startswith('--')
                ])
            ikwargs = copy.deepcopy(ikwargs)
            if 'help' in ikwargs:
                ikwargs['help'] = ikwargs['help'].format(
                    description=description,
                    prefix_arg=prefix_arg, prefix_dst=prefix_dst,
                    suffix_arg=suffix_arg, suffix_dst=suffix_dst,
                )
            if 'dest' in ikwargs:
                ikwargs['dest'] = ikwargs['dest'].format(
                    prefix_dst=prefix_dst,
                    suffix_dst=suffix_dst,
                )
            if use_flags:
                ikey = iargs[0]
            else:
                ikey = SubparserBase.arg2dest(iargs, ikwargs)
            out[ikey] = (iargs, ikwargs)
        if no_universal:
            return out
        for iargs, ikwargs in cls._arguments_universal:
            dest0 = SubparserBase.arg2dest(iargs, ikwargs)
            if dest0 in ignore:
                continue
            if use_flags:
                ikey = iargs[0]
            else:
                ikey = SubparserBase.arg2dest(iargs, ikwargs)
            out[ikey] = (iargs, ikwargs)
        return out

    @classmethod
    def any_arguments_set(cls, args, name=None, ignore=None,
                          no_universal=True):
        r"""Check if any arguments are set.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Name for the base variable.
            ignore (list, optional): Arguments to ignore.
            no_universal (bool, optional) If True, don't include
                universal arguments.

        Returns:
            bool: True if arguments are set.

        """
        for k in cls.argument_names(name, no_universal=no_universal):
            if ignore and k in ignore:
                continue
            if getattr(args, k, None) is not None:
                return True
        return False

    @classmethod
    def add_arguments(cls, dst, name=None):
        r"""Add composite arguments to a class.

        Args:
            dst (type): Class that arguments should be added to.
            name (str, optional): Name for the base variable. If not
                provided, all composite arguments will be added.

        """
        if name is None:
            if cls._name is None:
                for k in dst._composite_arguments.keys():
                    arg_class = get_class_registry().get('argument', k)
                    arg_class.add_arguments(dst)
                return
            for k in dst._composite_arguments[cls._name].keys():
                cls.add_arguments(dst, name=k)
            return
        assert cls._name is not None
        kwargs = dst._composite_arguments[cls._name][name]
        existing = dst.argument_dict()
        prefix = cls.get_prefix(name)
        suffix = cls.get_suffix(name)
        for k, v in cls.argument_dict(prefix=prefix, suffix=suffix,
                                      **kwargs).items():
            if k not in existing:
                dst._arguments.append(v)
        dst._argument_conversions.setdefault(cls._name, [])
        if name not in dst._argument_conversions[cls._name]:
            dst._argument_conversions[cls._name].append(name)

    @classmethod
    def is_date(cls, x):
        r"""Check if datetime instance is purely a date without time
        information.

        Args:
            x (datetime.datetime): Datetime instance to check.

        Returns:
            bool: True if x is purely a date.

        """
        if not isinstance(x, datetime):
            return False
        return all(getattr(x, k) == 0 for k in
                   ['hour', 'minute', 'second', 'microsecond'])

    @classmethod
    def to_date(cls, x):
        r"""Convert a datetime instance to a form that is purely a date.

        Args:
            x (datetime.datetime): Datetime instance to convert.

        Returns:
            datetime.datetime: Version with time information removed.

        """
        kws = {k: getattr(x, k) for k in
               ['year', 'month', 'day', 'tzinfo']}
        out = datetime(**kws)
        assert cls.is_date(out)
        return out

    @classmethod
    def parse(cls, x, args, name=None, **kwargs):
        r"""Parse an argument.

        Args:
            x (object): Instance to parse.
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Name that should be used to store the
                results.
            **kwargs: Additional keyword arguments are passed to the
                from_args method.

        Returns:
            object: The parsed instance.

        """
        args = copy.deepcopy(args)
        if name is None:
            name = cls._name
        cls.reset_args(name, args, value=x)
        return cls.from_args(name, args, **kwargs)

    def any_set(self, names):
        r"""Check if any of the specified arguments were set.

        Args:
            names (list): Argument names to check.

        Returns:
            bool: True if any of the arguments were set.

        """
        return any(self.args[k] is not None for k in names)

    def extract_unused(self, out, name):
        r"""Extract the equivalent value from an output.

        Args:
            out (object): Output.
            name (str): Argument to extract from out.

        Returns:
            object: Argument value.

        """
        raise NotImplementedError(name)

    def check_unused(self, names, output=NoDefault):
        r"""Assert that unused arguments were not set.

        Args:
            names (list): Argument names to check.
            output (object, optional): Output to try to extract arguments
                from.

        Raises:
            AssertionError: If unused arguments were set.

        """
        matches = {}
        invalid = []
        for k in names:
            if self.args[k] is not None:
                value = NoDefault
                if output is not NoDefault:
                    try:
                        value = self.extract_unused(output, k)
                        matches[k] = value
                    except NotImplementedError:
                        pass
                if value is NoDefault or self.args[k] != value:
                    invalid.append(k)
        if invalid:
            invalid = {
                self.prefix + k + self.suffix: self.args[k]
                for k in invalid
            }
            raise AssertionError(
                f'Unused arguments set:\n{pprint.pformat(invalid)}\n'
                f'Did not match:\n{pprint.pformat(matches)}'
            )

    def setdefaults(self, names):
        r"""Set defaults for missing arguments that were not set.

        Args:
            names (list): Argument names to set defaults for.

        Returns:
            bool: True if all defaults could be initialized, False
                otherwise.

        """
        out = True
        for k in names:
            value = None
            if k in self.ignore:
                assert self.args[k] is None
                out = False
                continue
            elif self.args[k] is not None:
                continue
            elif k in self.defaults:
                value = self.defaults[k]
            elif self.base and self.base.args[k] is not None:
                value = self.base.args[k]
            elif self.optional:
                out = False
                continue
            elif k in self._defaults:
                value = self._defaults[k]
            else:
                out = False
                continue
            self._defaults_set.append(k)
            self.args[k] = value
        return out

    def raw_args(self, name=None):
        r"""dict: Set of name/value pairs for arguments related to this
        argument."""
        if name is None:
            prefix = self.prefix
            suffix = self.suffix
        else:
            prefix = self.get_prefix(name)
            suffix = self.get_suffix(name)
        return {
            (prefix + k + suffix): v
            for k, v in self.args.items()
        }

    @property
    def value(self):
        r"""object: Parsed base argument value."""
        if hasattr(self, self._name):
            return getattr(self, self._name)
        if not self.setdefaults([self._name]):
            return None
        return self.args[self._name]

    @property
    def string(self):
        r"""str: String representation of this variable."""
        if self.is_wildcard(self._name):
            return '*'
        if not self.setdefaults([self._name]):
            return None
        if isinstance(self.args[self._name], str):
            return self.args[self._name]
        return None

    def is_wildcard(self, k):
        r"""Check if an argument is a wildcard.

        Args:
            k (str, list): Argument(s) to check.

        Returns:
            bool: True if argument is wildcard.

        """
        if isinstance(k, list):
            return any(self.is_wildcard(kk) for kk in k)
        return (self.args[k] == '*')

    def string_glob(self, wildcards):
        r"""Create a string with the provided list of parameters replaced
        with *.

        Args:
            wildcards (list): Arguments that should be replaced.

        Returns:
            str: Glob pattern.

        """
        if self._name in wildcards:
            return '*'
        args_overwrite = {k: '*' for k in wildcards}
        other = self.from_other(self, args_overwrite=args_overwrite)
        return other.string


class OutputArgument(CompositeArgument):
    r"""Container for output arguments."""

    _name = 'output'
    _defaults = {
        'output': False,
        'overwrite': False,
        'dont_write': False,
    }
    _arguments = [
        (('--output', ), {
            'nargs': '?', 'const': True,
            'help': (
                'File where {description} should be loaded from or '
                'saved to. If passed without a filename, the filename '
                'will be generated based on other arguments.'
            ),
        }),
        (('--overwrite', ), {
            'action': 'store_true',
            'help': (
                'Overwrite any existing {prefix_dst}output{suffix_dst} '
                'file generated or passed to '
                '\"--{prefix_arg}output{suffix_arg}\".'),
        }),
        (('--dont-write', ), {
            'action': 'store_true',
            'help': (
                'Don\'t write any output to '
                '{prefix_dst}output{suffix_dst} on disk (even if an '
                'explict path is provided). The generated output will '
                'still be available during the Python session.'
            ),
        }),
    ]
    _arguments_universal = [
        (('--overwrite-all', ), {
            'action': 'store_true',
            'help': 'Overwrite all child components of the task',
        }),
        (('--dont-write-all', ), {
            'action': 'store_true',
            'help': 'Don\'t write any output to disk',
        }),
    ]
    _attributes_kwargs = CompositeArgument._attributes_kwargs + [
        'ext', 'base', 'base_string', 'directory',
        'upstream', 'downstream', 'merge_all', 'merge_all_output',
    ]

    def __init__(self, name, args, ext=None, base=None,
                 base_string=None, directory=None,
                 upstream=None, downstream=None, merge_all=False,
                 merge_all_output=None, **kwargs):
        if upstream is None:
            upstream = []
        if downstream is None:
            downstream = {}
        self.ext = ext
        self.base_output = base
        self.base_string = base_string
        self._generated_path = None
        self.upstream = upstream
        self.downstream = downstream
        self._directory = directory
        self.merge_all = merge_all
        self.merge_all_output = merge_all_output
        self._uncached_args = {}
        self.output_name = self.get_suffix(name).strip('_')
        if self.base_string is None and self.base_output is None:
            self.base_string = self.output_name
        kwargs.setdefault('description', 'output')
        if 'output' not in kwargs.get('defaults', {}):
            kwargs.setdefault('defaults', {})
            if kwargs.get('optional', False):
                kwargs['defaults']['output'] = False
            else:
                kwargs['defaults']['output'] = True
        super(OutputArgument, self).__init__(name, args, **kwargs)

    def reset(self, args, **kwargs):
        r"""Reinitialize the arguments used by this instance.

        Args:
            args (argparse.Namespace): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to the
                base method.

        """
        super(OutputArgument, self).reset(args, **kwargs)
        self._uncached_args.clear()
        if self.base_output is not None:
            if isinstance(self.base_output, OutputArgument):
                base_output = getattr(
                    args, f'output_{self.base_output.output_name}')
                assert base_output is self.base_output
            else:
                base_output = getattr(args, f'output_{self.base_output}')
                self.base_output = base_output
            # If this fails it means that this output was initialized
            #   before its base or the base was initialized with a copy
            #   of args
            assert isinstance(self.base_output, OutputArgument)
        if self._directory is None:
            assert self.suffix
            self._uncached_args['directory'] = os.path.join(
                args.output_dir, self.output_name)
        self._uncached_args['id'] = getattr(args, 'id', None)
        if not self.generated:
            raise Exception(f'Filepath for \"{self.output_name}\" '
                            f'not generated: {self.path} '
                            f'{self.args["output"]}, '
                            f'{type(self.args["output"])}, '
                            f'{isinstance(self.args["output"], str)}')
        if self.generated:
            self.generate(args, reset=True)

    @property
    def value(self):
        r"""object: Parsed base argument value."""
        if not self.generated:
            return self.path
        return self.enabled

    @property
    def directory(self):
        r"""str: Directory where the generated file name should reside."""
        if self._directory is not None:
            return self._directory
        assert self._uncached_args['directory']
        return self._uncached_args['directory']

    @property
    def is_test(self):
        r"""bool: True if the output points to the test directory."""
        if isinstance(self.path, str):
            return self.path.startswith(cfg['directories']['test_output'])
        return False

    @cached_property
    def enabled(self):
        r"""str: Output file"""
        self.setdefaults(['output'])
        return bool(self.args['output'])

    @cached_property
    def is_unmerged(self):
        r"""bool: True if the output is an unmerged composite of multiple
        outputs."""
        return (self._uncached_args['id'] == 'all'
                and self.merge_all is not True)

    @cached_property
    def is_merged(self):
        r"""bool: True if the output is a merged composite of multiple
        outputs."""
        return ((self._uncached_args['id'] == 'all'
                 and self.merge_all is True)
                or self._uncached_args['id'] == self.merge_all)

    @cached_property
    def generated(self):
        r"""bool: True if the file name is generated."""
        self.setdefaults(['output'])
        return (not isinstance(self.args['output'], str))

    @cached_property
    def parts_generators(self):
        r"""dict: Generator methods for file name parts."""
        task = self.task_class
        out = {
            k: getattr(task, f'_output_{k}')
            for k in ['directory', 'base', 'suffix', 'ext']
            if hasattr(task, f'_output_{k}')
        }
        if 'base' not in out and self.base_string is None:
            assert self.base_output is not None
            out['base'] = self._base_generator
        return out

    @cached_property
    def parts_defaults(self):
        r"""dict: Defaults for file name parts."""
        assert self.directory
        out = {
            'directory': self.directory,
            'ext': self.ext,
            'base': self.base_string,
        }
        return out

    def assert_age_in_name(self, args):
        r"""Assert that the file name contains age information.

        Args:
            args (argparse.Namespace): Parsed arguments containing age.

        Raises:
            AssertionError: If the age string is not present.

        """
        if args.age.string == 'maturity' or not self.path:
            return
        if args.age.string not in self.path:
            raise AssertionError(f'\"{self.output_name}\" filename does '
                                 f'not contain age string '
                                 f'\"{args.age.string}\": {self.path}')

    @property
    def overwrite(self):
        r"""bool: True if overwrite set."""
        self.setdefaults(['overwrite_all', 'overwrite'])
        if self.args['overwrite_all']:
            return True
        return self.args['overwrite']

    def clear_overwrite(self, args=None):
        r"""Clear the overwrite parameter.

        Args:
            args (argparse.Namespace, optional): Parsed arguments to
               clear.

        """
        self.args['overwrite_all'] = False
        self.args['overwrite'] = False
        if args is not None:
            # setattr(args, 'overwrite_all', False)
            setattr(args, f'{self.prefix}overwrite{self.suffix}', False)

    @property
    def dont_write(self):
        r"""bool: True if dont_write set."""
        self.setdefaults(['dont_write_all', 'dont_write'])
        if self.args['dont_write_all']:
            return True
        return self.args['dont_write']

    @property
    def path(self):
        r"""str: File name."""
        if self.generated:
            if self._generated_path is not None:
                assert isinstance(self._generated_path, str)
            return self._generated_path
        assert isinstance(self.args['output'], (str, bool))
        return self.args['output']

    @property
    def exists(self):
        r"""bool: True if the file exists."""
        fname = self.path
        if self.generated and fname is True:
            raise Exception(f'{self.name} file not initialized')
        return isinstance(fname, str) and os.path.isfile(fname)

    def reset_generated(self, value=None):
        r"""Reset the generated path.

        Args:
            value (str, optional): New value for the generated path.

        """
        if value is False:
            value = None
        if value is not None:
            assert self.generated
            assert isinstance(value, str)
        self._generated_path = value

    def depends(self, arguments):
        r"""Check if the generated path depends on any of the listed
        arguments.

        Args:
            arguments (list): Set of arguments to check dependence on.

        Returns:
            bool: True if the path depends on any of the listed
                arguments.

        """
        if isinstance(arguments, str):
            arguments = [arguments]
        if not (self.generated and arguments):
            return False
        suffix = self.parts_generators.get('suffix', None)
        assert isinstance(suffix, SuffixGenerator)
        suffix_args = suffix.depends(self.output_name)
        if bool(set(suffix_args) & set(arguments)):
            return True
        base = self.base_output
        if base is None:
            return False
        return base.depends(arguments)

    @cached_property
    def task_class(self):
        r"""type: The task class that produces this output."""
        return TaskBase.get_output_task(self.output_name)

    @cached_property
    def default_args(self):
        r"""argparse.Namespace: Default arguments."""
        return self.task_class.copy_external_args(initialize=True)

    def remove_downstream(self, args, removed=None, wildcards=None,
                          args_copied=False, **kwargs):
        r"""Remove downstream outputs.

        Args:
            args (argparse.Namespace): Parsed arguments.
            removed (list, optional): List of outputs that have already
                been removed (to prevent duplication of effort).
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.
            args_copied (bool, optional): If True, the arguments have
                already been copied.
            **kwargs: Additional keyword arguments are passed to nested
                calls to 'remove' for downstream outputs.

        """
        if not self.downstream:
            return
        if removed is None:
            removed = []
        if wildcards is None:
            wildcards = []
        if self.is_merged or self.is_unmerged:
            wildcards = wildcards + ['id']
        task = self.task_class
        task.log_class(f"REMOVING DOWNSTREAM {self.output_name}:\n"
                       f"{pprint.pformat(self.downstream)}")
        wildcards = set(wildcards)
        args_base = set(task.argument_names())  # Use task for name?
        for ext, outputs in self.downstream.items():
            if all(k in removed for k in outputs):
                continue
            ext_wildcards = list(
                wildcards | (set(ext.argument_names()) - args_base)
            )
            ext_args = args
            if getattr(args, f'output_{ext._name}', None) is None:
                if args_copied:
                    ext.complete_external_args(ext_args)
                else:
                    ext_args = ext.copy_external_args(
                        args, initialize=True,
                    )
                    args_copied = True
            for k in outputs:
                if k in removed:
                    continue
                output = getattr(ext_args, f'output_{k}')
                output.remove(
                    args=ext_args, wildcards=ext_wildcards, **kwargs
                )
                removed.append(k)
                output.remove_downstream(
                    ext_args, removed=removed, wildcards=ext_wildcards,
                    args_copied=args_copied, **kwargs
                )

    def remove(self, args=None, wildcards=None,
               force=False, skip_test_output=False):
        r"""Remove the output file.

        Args:
            args (argparse.Namespace, optional): Parsed arguments.
                Only required if the path has not been set or wildcards
                are provided.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.
            force (bool, optional): If True, remove without asking for
                user input.
            skip_test_output (bool, optional): If True, don't include
                both test output and generated file name.

        """
        if self.is_merged or self.is_unmerged:
            if wildcards is None:
                wildcards = []
            wildcards = wildcards + ['id']
        if wildcards or self.path is None:
            assert args is not None
            fname = self.generate(args, wildcards=wildcards)
        else:
            fname = self.path
        if not fname:
            return
        fnames = [fname]
        if not skip_test_output:
            if fname.startswith(cfg['directories']['output']):
                fnames.append(fname.replace(
                    cfg['directories']['output'],
                    cfg['directories']['test_output']
                ))
            elif fname.startswith(cfg['directories']['test_output']):
                fnames.append(fname.replace(
                    cfg['directories']['test_output'],
                    cfg['directories']['output']
                ))
        files = []
        for x in fnames:
            files += glob.glob(x)
        if not files:
            return
        if not force:
            if not utils.input_yes_or_no(
                    f'Remove existing \"{self.output_name}\" '
                    f'output(s):\n{pprint.pformat(files)}?'):
                return
        for x in files:
            os.remove(x)

    def _base_generator(self, args, name, wildcards=None):
        assert name == self.output_name
        base = self.base_output
        assert base is not None
        return base.generate(args, wildcards=wildcards)

    def generate(self, args, reset=False, wildcards=None):
        r"""Generate the file name.

        Args:
            args (argparse.Namespace): Parsed arguments that file name
                should be generated from.
            reset (bool, optional): If True, reset the generated path
                on the instance.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.

        Returns:
            str: Generated file name.

        """
        task = self.task_class
        assert self.generated
        assert self.output_name in task._outputs_local
        try:
            fname = self._generate(args, wildcards=wildcards)
        except FilenameGenerationError as e:
            task.log_class(f"FILENAME GENERATION ERROR: {task} {e}")
            self.args['dont_write'] = True
            fname = False
        if reset:
            assert not wildcards
            self.reset_generated(value=fname)
            # task.log_class(f'Generated \"{self.output_name}\" '
            #                f'path: {self.path}')
        return fname

    def _generate(self, args, wildcards=None):
        iparts = {}
        for k, v in self.parts_generators.items():
            if k in iparts:
                continue
            iparts[k] = v(args, self.output_name, wildcards=wildcards)
            if iparts[k] is None:
                del iparts[k]
        for k, v in self.parts_defaults.items():
            if k not in iparts:
                iparts[k] = v
        disabled = [k for k, v in iparts.items() if v is False]
        if disabled:
            raise FilenameGenerationError(
                f'\"{self.output_name}\" output disabled by '
                f'\"{disabled}\"'
            )
        base = iparts.pop('base')
        fname = utils.generate_filename(base, **iparts)
        if wildcards:
            while '**' in fname:
                fname = fname.replace('**', '*')
        return fname


class ColorArgument(CompositeArgument):
    r"""Container for color arguments."""

    _name = 'color'
    _defaults = {
        'color': 'transparent',
    }
    _arguments = [
        (('--color', ), {
            'type': utils.parse_color,
            'help': (
                'Color name or tuple of RGBA values expressed as float '
                'in the range [0, 1]{description}.'
            ),
        }),
    ]

    @cached_property
    def color(self):
        r"""tuple: RGBA color tuple."""
        if not self.setdefaults(['color']):
            return None
        return utils.parse_color(self.args['color'], convert_names=True)


class AxisArgument(CompositeArgument):
    r"""Container for parsing an axis argument."""

    _name = 'axis'
    _arguments = [
        (('--axis', ), {
            'type': 'str',
            'help': 'Name or vector{description}',
        }),
    ]

    @property
    def string(self):
        r"""str: String representation of this variable."""
        if not self.setdefaults(['axis']):
            return None
        if self.args['axis'] in utils._axis_map:
            return self.args['axis']
        return None

    @cached_property
    def value(self):
        r"""object: Parsed base argument value."""
        if not self.setdefaults(['axis']):
            return None
        return utils.parse_axis(self.args['axis'])


# class AxesArgument(CompositeArgument):
#     r"""Container for parsing axes arguments."""

#     _name = 'axes'
#     _defaults = {
#         'up': 'y',
#         'north': 'x',
#         'east': 'z',
#     }
#     _arguments = [
#         (('--axis-up', ), {
#             'type': AxisArgument,
#             'help': (
#                 'Axis name or vector for up direction{description}.'
#             ),
#         }),
#         (('--axis-east', ), {
#             'type': AxisArgument,
#             'help': (
#                 'Axis name or vector for east direction{description}.'
#             ),
#         }),
#         (('--axis-north', ), {
#             'type': AxisArgument,
#             'help': (
#                 'Axis name or vector for north direction{description}.'
#             ),
#         }),
#     ]

#     # def __init__(self, name, args, **kwargs):
#     #     super(AxesArgument, self).__init__(name, args, **kwargs)
#     #     if self.args['axis_up']


# class CameraArgument(CompositeArgument):
#     r"""Container for parsing camera arguments."""

#     _name = 'camera'
#     _name_as_prefix = True
#     _defaults = {
#         'lens': 'projection',
#         'direction': 'downsoutheast',
#         'fov_width': parse_quantity(45.0, 'degrees'),
#         'fov_height': parse_quantity(45.0, 'degrees'),
#     }
#     _arguments = [
#         (('--lens', ), {
#             'type': str,
#             'choices': ['projection', 'orthographic'],  # 'spherical'],
#             'help': 'Type of camera{description}',
#         }),
#         (('--direction', ), {
#             'type': str,
#             'help': (
#                 'Direction that camera should face. If not '
#                 'provided, the camera will point to the center of '
#                 'the scene from its location.',
#             ),
#         }),
#         (('--fov-width', ), {
#             'type': parse_quantity, 'units': 'degrees',
#             'help': (
#                 'Angular width of the camera\'s field of view (in '
#                 'degrees) for a projection camera.'
#             ),
#         }),
#         (('--fov-height', ), {
#             'type': parse_quantity, 'units': 'degrees',
#             'help': (
#                 'Angular height of the camera\'s field of view (in '
#                 'degrees) for a projection camera.'
#             ),
#         }),
#         (('--up', ), {
#             'type': str,
#             'help': (
#                 'Up direction for the camera. If not provided, the '
#                 'up direction for the scene will be assumed.'
#             ),
#         }),
#         (('--location', ), {
#             'type': str,
#             'help': (
#                 'Location of the camera. If not provided, one will '
#                 'be determined that captures the entire scene from '
#                 'the provided camera direction. If a direction is '
#                 'also not provided, the camera will be centered '
#                 'on the center of the scene facing down, '
#                 'southeast at a distance that captures the entire '
#                 'scene. If \"maturity\" is specified, the location will '
#                 'be set for the mature plant (only valid for generated '
#                 'meshes).'
#             ),
#         }),
#     ]

#     def __init__(self, name, args, **kwargs):
#         super(CameraArgument, self).__init__(name, args, **kwargs)
#         if ((self.args['direction'] is None
#              and self.args['location'] is None)):
#             self.setdefaults(['direction'])
#         # TODO

#     @cached_property
#     def direction(self):
#         r"""np.ndarray: Unit vector for camera's pointing direction."""
#         if self.args['direction'] is not None:
#             return


class AgeArgument(CompositeArgument):
    r"""Container for parsing age arguments."""

    _name = 'age'
    _defaults = {
        'age': 'maturity',
    }
    _arguments = [
        (('--age', ), {
            'type': parse_quantity, 'units': 'days',
            'help': 'Plant age (in days since planting){description}.',
        }),
    ]
    _attributes_copy = [
        '_parameter_inst', '_parameter_inst_function',
    ]

    def __init__(self, name, args, **kwargs):
        self._time_argument = None
        self._parameter_inst = None
        self._parameter_inst_function = None
        super(AgeArgument, self).__init__(name, args, **kwargs)

    def reset(self, args, **kwargs):
        r"""Reinitialize the arguments used by this instance.

        Args:
            args (argparse.Namespace): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to the
                base method.

        """
        super(AgeArgument, self).reset(args, **kwargs)
        self._parameter_inst = None
        self._parameter_inst_function = getattr(
            args, '_parameter_inst_function', None)

    @classmethod
    def is_crop_age(cls, x):
        r"""Check if a string is a named crop age.

        Args:
            x (str): Value to check.

        Returns:
            bool: True if x is a named crop age.

        """
        from canopy_factory.crops import ParametrizeCropTask
        return (x in ParametrizeCropTask._age_strings)

    @property
    def parameter_inst(self):
        r"""ParametrizeCropTask: Instance for calculating ages."""
        if ((self._parameter_inst is None
             and self._parameter_inst_function is not None)):
            self._parameter_inst = self._parameter_inst_function()
        assert self._parameter_inst is not None
        return self._parameter_inst

    @property
    def requires_parameter_inst(self):
        r"""bool: True if a parameter instance is required."""
        return self.is_crop_age(self.args['age'])

    def get_crop_age(self, x, return_quantity=False):
        r"""Get a named crop age as a timedelta.

        Args:
            x (str): Named crop age.

        Returns:
            timedelta: Age as a time delta.

        """
        out = self.parameter_inst.get_age(x)
        if return_quantity:
            return out
        return utils.quantity2timedelta(out)

    @cached_property
    def crop_age_string(self):
        r"""str: Crop-based age string."""
        self.setdefaults(['age'])
        if self.is_crop_age(self.args['age']):
            return self.args['age']
        return None

    @cached_property
    def string(self):
        r"""str: String representation of the age."""
        if self.is_wildcard('age'):
            return '*'
        if self.crop_age_string:
            return self.crop_age_string
        if self.age is None:
            return None
        return f'{int(self.age.to("days"))}days'

    @cached_property
    def age(self):
        r"""units.Quantity: Days since planting."""
        if self._time_argument is not None:
            return self._time_argument.age
        if not self.setdefaults(['age']):
            return None
        if self.is_crop_age(self.args['age']):
            return self.get_crop_age(self.args['age'],
                                     return_quantity=True)
        return self.args['age']


class TimeArgument(AgeArgument):
    r"""Container for parsing time arguments."""

    _name = 'time'
    _defaults = dict(
        AgeArgument._defaults,
        **{
            'hour': 'noon',
            'timezone': pytz.timezone("America/Chicago"),
            # 'doy': 169,  # 06/17
            'doy': 173,  # 06/21
            'year': 2024,
            'location': 'Champaign',
            'latitude': parse_quantity(40.1164, 'degrees'),
            'longitude': parse_quantity(-88.2434, 'degrees'),
            'temperature': parse_quantity(12.0, 'degC'),
            'altitude': parse_quantity(224.0, 'meters'),
        }
    )
    _arguments = AgeArgument._arguments + [
        (('--time', '-t'), {
            'type': str,
            'help': (
                'Date time (in any ISO 8601 format){description}.'
                'If time information is not provided, the provided '
                '\"--{prefix_arg}hour{suffix_arg}\" will be used. '
                'If \"now\" is specified the current date and time '
                'will be used. If \"sunrise\", \"noon\", \"transit\", '
                'or \"sunset\" is specified, the time will be calculated'
                ' from the provided date and location.'
            ),
        }),
        (('--hour', '--hr', ), {
            'type': int,
            'help': (
                'Hour{description}. If provided with '
                '\"--{prefix_arg}time{suffix_arg}\", any hour '
                'information in the specified time will be overwritten. '
                'Defaults to 12 if \"--{prefix_arg}doy{suffix_arg}\" is '
                'provided, but \"--{prefix_arg}hour{suffix_arg}\" is '
                'not.'
            ),
        }),
        (('--date', ), {
            'type': str,
            'help': (
                'Date that should be used with '
                '\"--{prefix_arg}time{suffix_arg}\" '
                'if a string is provided describing the time of day '
                '(e.g. \"sunrise\", \"noon\", \"sunset\"). If '
                'provided, \"--{prefix_arg}doy{suffix_arg}\" and '
                '\"--{prefix_arg}year{suffix_arg}\" will not be used. '
            ),
        }),
        (('--doy', ), {
            'type': int,
            'help': 'Day of the year{description}.'
        }),
        (('--year', ), {
            'type': int,
            'help': (
                'Year{description}. If provided '
                'with \"--{prefix_arg}time{suffix_arg}\", '
                'the year in the time string(s) '
                'will be overwritten. Defaults to the current year '
                'if \"--{prefix_arg}doy{suffix_arg}\" is provided, '
                'but \"--{prefix_arg}year{suffix_arg}\" is not.'
            ),
        }),
    ]
    _arguments_universal = AgeArgument._arguments_universal + [
        (('--planting-date', ), {
            'type': str,
            'help': (
                'Date time (in any ISO 8601 format) on which the crop '
                'was planted (used to calculate age from time or time '
                'from age.'
            ),
        }),
        (('--location', ), {
            'type': str,
            'choices': sorted(list(utils.read_locations().keys())),
            'help': ('Name of a registered location that should be used '
                     'to set the location dependent properties: '
                     'timezone, altitude, longitude, latitude'),
        }),
        (('--timezone', '--tz', ), {
            'type': pytz.timezone,
            'help': (
                'Name of timezone (as accepted by pytz){description}. '
                'If provided '
                'with \"--{prefix_arg}time{suffix_arg}\", '
                'any timezone information in the '
                'specified time(s) will be overwritten. Defaults '
                'to \"America/Chicago\" if \"--doy\" is provided, '
                'but \"--timezone\" is not.'
            ),
        }),
        (('--latitude', '--lat', ), {
            'type': parse_quantity, 'units': 'degrees',
            'help': ('Latitude (in degrees) at which the sun should be '
                     'modeled. Defaults to the latitude of Champaign '
                     'IL.'),
        }),
        (('--longitude', '--long', ), {
            'type': parse_quantity, 'units': 'degrees',
            'help': ('Longitude (in degrees) at which the sun should be '
                     'modeled. Defaults to the longitude of Champaign '
                     'IL.'),
        }),
        (('--altitude', '--elevation', ), {
            'type': parse_quantity, 'units': 'meters',
            'help': ('Altitude (in meters) that should be used for '
                     'solar light calculations. If not provided, it '
                     'will be calculated from \"pressure\", if it is '
                     'provided, and the elevation of Champaign, IL '
                     'otherwise.'),
        }),
        (('--pressure', ), {
            'type': parse_quantity, 'units': 'Pa',
            'help': ('Air pressure (in Pa) that should be used for '
                     'solar light calculations. If not provided, it '
                     'will be calculated from \"altitude\".'),
        }),
        # TODO: This should depend on time
        (('--temperature', ), {
            'type': parse_quantity, 'units': 'degC',
            'help': ('Air temperature (in degrees C) that should be '
                     'used for solar light calculations.'),
        }),
    ]
    _attributes_copy = AgeArgument._attributes_copy + [
        'ignore_date',
    ]

    def __init__(self, name, args, **kwargs):
        self.ignore_date = False
        super(TimeArgument, self).__init__(name, args, **kwargs)

    def reset(self, args, **kwargs):
        r"""Reinitialize the arguments used by this instance.

        Args:
            args (argparse.Namespace): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to the
                base method.

        """
        super(TimeArgument, self).reset(args, **kwargs)
        if 'location' not in self.ignore:
            self.setdefaults(['location'])
            if ((isinstance(self.args['location'], str)
                 and not self.is_wildcard('location'))):
                arguments = self.argument_dict()
                location_data = utils.read_locations()[
                    self.args['location']]
                for k, v in location_data.items():
                    if k == 'name':
                        continue
                    assert k in arguments
                    if 'units' in arguments[k][1]:
                        self.args[k] = parse_quantity(
                            v, arguments[k][1]['units'])
                    else:
                        self.args[k] = v
        if self.is_solar_time(self.args['time']):
            assert self.args['hour'] is None
            self.args['hour'] = self.args['time']
            self.args['time'] = None
        elif self.args['time'] == 'now':
            self.args['time'] = datetime.now()
        elif ((isinstance(self.args['time'], str)
               and not self.is_wildcard('time'))):
            self.args['time'] = datetime.fromisoformat(self.args['time'])
        if self.is_date(self.args['time']):
            assert self.args['date'] is None
            self.args['date'] = self.args['time']
            self.args['time'] = None
        if self.args['date'] == 'now':
            self.args['date'] = self.to_date(datetime.now())
        elif self.is_crop_age(self.args['date']):
            assert 'age' not in self.ignore
            assert self.args['age'] is None or 'age' in self._defaults_set
            self.args['age'] = self.args['date']
            if self.args['age'] == 'planting':
                self.defaults['hour'] = 'noon'
            self.args['date'] = None
        if ((isinstance(self.args['timezone'], str)
             and not self.is_wildcard('timezone'))):
            self.args['timezone'] = pytz.timezone(self.args['timezone'])
        if ((self.args['age'] is None
             and self.args['planting_date'] is None)):
            self.setdefaults(['age'])
        if (('planting_date' not in self.ignore
             and isinstance(self.args['planting_date'], str)
             and not self.is_wildcard('planting_date'))):
            self.args['planting_date'] = datetime.fromisoformat(
                self.args['planting_date'])
            assert self.is_date(self.args['planting_date'])
        if ((self.crop_age_string == 'planting'
             and self.args['planting_date'] is None)):
            self.args['planting_date'] = self.date

    @classmethod
    def reset_args(cls, name, args, value=NoDefault):
        r"""Reset the arguments for a variable.

        Args:
            name (str): Variable name.
            args (argparse.Namespace): Parsed arguments.
            value (object, optional): Value that the parsed variable
                should be reset to.

        """
        age_value = NoDefault if value is NoDefault else None
        super(TimeArgument, cls).reset_args(name, args, value=value)
        prefix = cls.get_prefix(name)
        suffix = cls.get_suffix(name)
        AgeArgument.reset_args(prefix + 'age' + suffix, args,
                               value=age_value)

    def update_args(self, args, name=None):
        r"""Update a namespace with the parsed time arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Name that should be used for the set
                variables.

        """
        if name is None:
            name = self.name
            prefix = self.prefix
            suffix = self.suffix
        else:
            prefix = self.get_prefix(name)
            suffix = self.get_suffix(name)
        super(TimeArgument, self).update_args(args, name=name)
        if 'age' not in self.ignore:
            age_inst = AgeArgument.from_other(self)
            age_inst._time_argument = self
            setattr(args, f'{prefix}age{suffix}', age_inst)

    def extract_unused(self, out, name):
        r"""Extract the equivalent value from an output.

        Args:
            out (object): Output.
            name (str): Argument to extract from out.

        Returns:
            object: Argument value.

        """
        if isinstance(out, datetime):
            if name in ['year', 'hour']:
                return getattr(out, name)
            elif name == 'timezone':
                return out.tzinfo
            elif name == 'doy':
                return int(out.strftime('%j'))
            elif name == 'date':
                return self.to_date(out)
        return super(TimeArgument, self).extract_unused(out, name)

    @property
    def is_summer_solstice(self):
        r"""bool: True if the date is the summer solstice."""
        if self.args['latitude'] > units.Quantity(0, 'degrees'):
            return (self.date.month == 6 and self.date.day == 21)
        return (self.date.month == 12 and self.date.day == 21)

    @property
    def is_summer_solstice_noon(self):
        r"""bool: True if the time is noon on the summer solstice."""
        return (self.is_summer_solstice
                and self.solar_time_string in ['noon', 'transit'])

    @property
    def summer_solstice(self):
        r"""datetime: Get the date of the summer solstice for this
        latitude."""
        if self.args['latitude'] > units.Quantity(0, 'degrees'):
            return self.date.replace(month=6, day=21)
        return self.date.replace(month=12, day=21)

    def iteration_args(self, dt=None, include_bookends=False,
                       dont_age=False):
        r"""Arguments that should be passed to represent this time
        in an iteration.

        Args:
            dt (units.Quantity, optional): Time interval to apply.
            include_bookends (bool, optional): If True, the keys in the
                returned arguments should include the prefix & suffix.
            dont_age (bool, optional): If True, don't change the age.

        Returns:
            dict: Arguments.

        """
        if dt is None:
            dt = units.Quantity(0.0, 'days')
        dt_null = (dt == units.Quantity(0.0, 'days'))
        dt_days = ((dt % units.Quantity(1.0, 'days'))
                   == units.Quantity(0.0, 'days'))
        out = {
            'time': self.time + utils.quantity2timedelta(dt),
            'date': None,
            'age': None,
            'planting_date': None,
        }
        for k in self.argument_names(no_universal=True):
            out.setdefault(k, None)
        if self.solar_time_string and (dt_null or dt_days):
            out.update(
                time=self.solar_time_string,
                date=(self.date + utils.quantity2timedelta(dt)),
            )
        if self.crop_age_string and (dt_null or dont_age):
            out.update(
                age=self.crop_age_string,
            )
        elif dont_age:
            out.update(
                age=self.age,
            )
        elif self.args['age'] is not None:
            out.update(
                age=(self.age + dt),
            )
        else:
            out.update(
                planting_date=self.planting_date,
            )
        if not include_bookends:
            return out
        return {f'{self.prefix}{k}{self.suffix}': v
                for k, v in out.items()}

    @classmethod
    def is_solar_time(cls, x):
        r"""Check if a string is a named solar time.

        Args:
            x (str): Time to check.

        Returns:
            bool: True if x is a named solar time.

        """
        return (x in utils.SolarModel._solar_times)

    @cached_property
    def solar_time_string(self):
        r"""str: Solar time string."""
        if self.args['time'] is not None:
            return None
        if not self.setdefaults(['hour']):
            return None
        if self.is_solar_time(self.args['hour']):
            return self.args['hour']
        return None

    @cached_property
    def string(self):
        r"""str: String representation of the time."""
        out = ''
        if ((self.args['location'] is not None
             and self.args['location'] != self._defaults['location'])):
            out += f"{self.args['location']}-"
        if self.is_wildcard(['time', 'date', 'hour', 'year',
                             'timezone']):
            out += '*'
            return out
        if self.time is None:
            return None
        x = self.time
        if self.args['location'] is not None:
            x = x.replace(tzinfo=None)
        if self.solar_time_string:
            if not self.ignore_date:
                date = self.to_date(x)
                out += date.date().isoformat() + '-'
            out += self.solar_time_string
            return out
        assert not self.ignore_date  # TODO
        out += x.replace(microsecond=0).isoformat().replace(':', '-')
        return out

    @cached_property
    def date(self):
        r"""datetime.datetime: Parsed date instance."""
        if self.args['time'] is not None:
            return self.to_date(self.time)
        elif self.args['date'] is not None:
            if isinstance(self.args['date'], str):
                self.args['date'] = datetime.fromisoformat(
                    self.args['date'])
            out = self.args['date']
            assert self.is_date(out)
            self.check_unused(['year', 'doy'], out)
            return out
        elif ((self.args['planting_date'] is not None
               and self.args['age'] is not None)):
            out = self.planting_date + utils.quantity2timedelta(self.age)
            out = self.to_date(out)
            self.check_unused(['year', 'doy'], out)
            return out
        elif self.base and not self.any_set(['year', 'doy']):
            return copy.deepcopy(self.base.date)
        if not self.setdefaults(['year', 'doy']):
            return None
        return datetime.strptime(
            f"{self.args['year']}-{self.args['doy']}",
            "%Y-%j"
        )

    @cached_property
    def time(self):
        r"""datetime.datetime: Parsed time instance."""
        if self.args['time'] is not None:
            if self.args['time'].tzinfo is None:
                self.setdefaults(['timezone'])
                self.args['time'] = self.args['time'].astimezone(
                    self.args['timezone'])
            else:
                self.setdefaults(['timezone'])
                x = self.args['time'].astimezone(self.args['timezone'])
                assert x == self.args['time']
                self.args['time'] = x
            out = self.args['time']
            self.check_unused(['date', 'hour', 'year', 'doy'], out)
            return out
        date = self.date
        if date is None or not self.setdefaults(['timezone', 'hour']):
            return None
        date = date.astimezone(self.args['timezone'])
        if self.solar_time_string:
            solar_model = self._get_solar_model(date)
            return solar_model.solar_time(self.solar_time_string)
        return date.replace(hour=self.args['hour'])

    @cached_property
    def age(self):
        r"""units.Quantity: Days since planting."""
        if self.args['age'] is not None:
            if self.is_crop_age(self.args['age']):
                return self.get_crop_age(self.args['age'],
                                         return_quantity=True)
            return self.args['age']
        if self.args['planting_date'] is None:
            return None
        if self.date is None:
            return None
        diff = self.date - self.planting_date
        return utils.timedelta2quantity(diff)

    @cached_property
    def planting_date(self):
        r"""datetime.datetime: Planting date."""
        if self.args['planting_date'] is not None:
            return self.args['planting_date']
        if self.time is None or self.age is None:
            return None
        out = self.time - utils.quantity2timedelta(self.age)
        return self.to_date(out)

    def _get_solar_model(self, time):
        if not (self.args['altitude'] or self.args['pressure']):
            self.setdefaults(['altitude'])
        self.setdefaults(['latitude', 'longitude', 'temperature'])
        return utils.SolarModel(
            self.args['latitude'], self.args['longitude'], time,
            altitude=self.args['altitude'],
            pressure=self.args['pressure'],
            temperature=self.args['temperature'],
        )

    @cached_property
    def solar_model(self):
        r"""SolarModel: Solar model."""
        if self.time is None:
            return None
        return self._get_solar_model(self.time)


class RepeatIterationError(BaseException):
    r"""Error can be raised if a step should be repeated."""

    def __init__(self, args_overwrite=None):
        if args_overwrite is None:
            args_overwrite = {}
        self.args_overwrite = args_overwrite
        super(RepeatIterationError, self).__init__()


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
    _arguments = []
    _argument_conversions = {}
    _external_arg_attributes = []
    _subparser_arguments = {}
    _subparser_modifications = {}
    _composite_arguments = {}

    @staticmethod
    def _modify_argument(arg, keys=None, append_choices=None, **kwargs):
        flags, attr = arg[:]
        attr = dict(attr, **kwargs)
        if keys:
            if isinstance(keys, str):
                keys = (keys, )
            flags = tuple(list(flags) + list(keys))
        if append_choices:
            if isinstance(append_choices, str):
                append_choices = [append_choices]
            attr.setdefault('choices', [])
            attr['choices'] += append_choices
        return (flags, attr)

    @classmethod
    def add_arguments(cls, parser, only_subparser=False,
                      exclude=None, include=None,
                      modifications=None):
        r"""Add arguments associated with this subparser to a parser.

        Args:
            parser (InstrumentedParser): Parser that the arguments
                should be added to.
            only_subparser (bool, optional): If True, only add the
                subparser if it is missing.
            exclude (list, optional): Set of arguments to exclude.
            include (list, optional): Set of arguments to include.
            modifications (dict, optional): Modifications that should be
                made to arguments.

        """
        add_missing = {'help': cls._help}
        add_missing_group = {}
        if cls._default is not None:
            add_missing_group['default'] = cls._default
        subparser = parser.get_subparser(
            cls._registry_key, cls._name,
            add_missing_group=add_missing_group,
            add_missing=add_missing,
        )
        if modifications is None:
            modifications = {}

        def check(k):
            if exclude is not None and k in exclude:
                return False
            if include is not None and k not in include:
                return False
            return True

        for k in cls._subparser_arguments.keys():
            if not check(k):
                continue
            for v in get_class_registry().values(k):
                v.add_arguments(subparser, only_subparser=True,
                                exclude=exclude, include=include,
                                modifications=modifications)
        if only_subparser:
            return
        for iargs, ikwargs in cls._arguments:
            idest = ikwargs.get(
                'dest', iargs[0].lstrip('-').replace('-', '_'))
            if not check(idest):
                continue
            if idest in modifications:
                iargs, ikwargs = SubparserBase._modify_argument(
                    (iargs, ikwargs), **modifications[idest])
            try:
                subparser.add_argument(*iargs, **ikwargs)
            except argparse.ArgumentError:
                print(cls, subparser)
                pprint.pprint(cls._arguments)
                pdb.set_trace()
                raise
        for k in cls._subparser_arguments.keys():
            if not check(k):
                continue
            kinclude = include
            kmodifications = modifications
            if isinstance(cls._subparser_arguments[k], list):
                kinclude = [
                    x for x in cls._subparser_arguments[k] if check(x)
                ]
            if k in cls._subparser_modifications:
                kmodifications = dict(cls._subparser_modifications[k],
                                      **modifications)
            for v in get_class_registry().values(k):
                v.add_arguments(subparser, only_subparser=False,
                                exclude=exclude, include=kinclude,
                                modifications=kmodifications)

    @classmethod
    def _convert_composite(cls, args, k, v=NoDefault, base=None,
                           overwrite=False, **kwargs):
        if base is None:
            for kk, vv in cls._composite_arguments.items():
                if k in vv:
                    base = kk
                    break
            else:
                raise ValueError(f'Could not determine composite '
                                 f'argument type for \"{k}\"')
        arg_class = get_class_registry().get('argument', base)
        if overwrite:
            arg_class.reset_args(k, args, value=v)
        kws = dict(
            cls._composite_arguments.get(base, {}).get(k, {}), **kwargs
        )
        # if base == 'output':
        #     kws['dont_reset'] = True
        # if base == 'output' and 'base' in kws:
        #     # Ensure that the base output is initialized first
        #     assert not overwrite  # TODO: Should this overwrite base?
        #     base_cls = cls._get_output_task(cls, kws['base'])
        #     base_cls._convert_composite(args, f"output_{kws['base']}",
        #                                 base=base, overwrite=overwrite)
        return arg_class.from_args(k, args, **kws)

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
    def arg2default(cls, name, info=None):
        r"""Determine the default value for an argument.

        Args:
            name (str): Argument name.
            info (dict, optional): Argument info.

        Returns:
            object: Default value for the argument.

        """
        ival = None
        if info is None:
            info = cls.argument_dict(include=name,
                                     include_subparser='explicit')[1]
        if info.get('action', None) == 'store_true':
            ival = False
        elif info.get('action', None) == 'store_false':
            ival = True
        elif 'default' in info:
            ival = info['default']
            if 'type' in info and ival is not None:
                ival = info['type'](ival)
        return ival

    @classmethod
    def add_missing_args(cls, args):
        r"""Add missing arguments.

        Args:
            args (argparse.Namespace): Parsed arguments to modify.

        """
        for k, v in cls.argument_dict(
                include_subparser='explicit').items():
            if not hasattr(args, k):
                setattr(args, k, cls.arg2default(k, info=v[1]))

    @classmethod
    def argument_dict(cls, exclude=None, include=None,
                      modifications=None, only_subparser=False,
                      include_subparser=False, use_flags=False):
        r"""Get a dictionary of argument data.

        Args:
            exclude (list, optional): Set of arguments to exclude.
            include (list, optional): Set of arguments to include.
            modifications (dict, optional): Modifications that should be
                made to arguments.
            only_subparser (bool, optional): If True, include only
                subparser arguments.
            include_subparser (bool, optional): If True, include
                subparser arguments. If 'explicit' is passed, the
                subparser itself will also be added as an argument.
            use_flags (bool, optional): If False, use the argument
                destinations. If True, use the argument flags.

        Returns:
            dict: Positional and keyword arguments for creating the
                arguments.

        """
        out = OrderedDict()
        return_first = False
        if isinstance(include, str):
            return_first = True
            include = [include]
        if modifications is None:
            modifications = {}
        if only_subparser and not include_subparser:
            include_subparser = True

        def check(k):
            if exclude is not None and k in exclude:
                return False
            if include is not None and k not in include:
                return False
            return True

        def check_and_add(iargs, ikwargs):
            if return_first and out:
                return True
            idst = cls.arg2dest(iargs, ikwargs)
            if not check(idst):
                return False
            if include is not None and idst not in include:
                return False
            if idst in modifications:
                iargs, ikwargs = SubparserBase._modify_argument(
                    (iargs, ikwargs), **modifications[idst])
            if use_flags:
                ikey = iargs[0]
            else:
                ikey = idst
            if ikey not in out:
                if return_first and include[0] != idst:
                    return False
                out[ikey] = (iargs, ikwargs)
                if return_first:
                    return True
            return False

        if include_subparser == 'explicit':
            k = cls._registry_key
            iargs = (k, )
            ikwargs = {'choices': get_class_registry().keys(k)}
            vbase = get_class_registry().getbase(k)
            if vbase._default:
                ikwargs['default'] = vbase._default
            if vbase._help:
                ikwargs['help'] = vbase._help
            if check_and_add(iargs, ikwargs):
                pass
            for k in cls._subparser_arguments.keys():
                if return_first and out:
                    break
                if cls._subparser_arguments[k] is False or not check(k):
                    continue
                for v in get_class_registry().values(k):
                    if return_first and out:
                        break
                    for iargs, ikwargs in v.argument_dict(
                            only_subparser=True,
                            include_subparser=include_subparser,
                            exclude=exclude, include=include,
                            modifications=modifications,
                            use_flags=use_flags).values():
                        if check_and_add(iargs, ikwargs):
                            break
        if not only_subparser:
            for iargs, ikwargs in cls._arguments:
                if check_and_add(iargs, ikwargs):
                    break
        if include_subparser:
            for k, kargs in cls._subparser_arguments.items():
                if return_first and out:
                    break
                if kargs is False or not check(k):
                    continue
                kinclude = include
                kmodifications = modifications
                if isinstance(kargs, list):
                    kinclude = [x for x in kargs if check(x)]
                if k in cls._subparser_modifications:
                    kmodifications = dict(cls._subparser_modifications[k],
                                          **modifications)
                for v in get_class_registry().values(k):
                    if return_first and out:
                        break
                    for iargs, ikwargs in v.argument_dict(
                            only_subparser=False,
                            include_subparser=True,
                            exclude=exclude, include=kinclude,
                            modifications=kmodifications,
                            use_flags=use_flags).values():
                        if check_and_add(iargs, ikwargs):
                            break
        if return_first:
            if not out:
                raise ValueError(
                    f'{cls} Could not find argument matching '
                    f'\"{include[0]}\":\n{pprint.pformat(out)}')
            assert len(out) == 1
            return list(out.values())[0]
        return out

    @staticmethod
    def argument_names_static(cls, use_flags=False):
        r"""Set of arguments allowed by the subparser.

        Args:
            use_flags (bool, optional): If False, use the argument
                destinations. If True, use the argument flags.

        Returns:
            list: Arguments.

        """
        # TODO: Prepend with subparser registry key?
        out = list(cls._subparser_arguments.keys())
        for k in cls._subparser_arguments.keys():
            out += [
                x for x in cls._get_subparser_arguments(
                    k, use_flags=use_flags)
                if x not in out
            ]
        for iargs, ikwargs in cls._arguments:
            if use_flags:
                out.append(iargs[0])
            elif issubclass(cls, SubparserBase):
                out.append(cls.arg2dest(iargs, ikwargs))
            else:
                out.append(SubparserBase.arg2dest(iargs, ikwargs))
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
        return cls.argument_names_static(cls, use_flags=use_flags)

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

    def run(self):
        r"""Run the process associated with this subparser."""
        raise NotImplementedError


class FilenameGenerationError(BaseException):
    r"""Error to raise when a file name cannot be generated."""
    pass


class SuffixGenerationError(FilenameGenerationError):
    r"""Error to raise when an argument cannot be converted into a
    suffix."""
    pass


class ArgumentSuffix(object):
    r"""Class for generating a suffix from an argument.

    Args:
        name (str, list): Argument name(s).
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
        composite (str, optional): Name of the composite argument type
            that will be used to parse this argument.
        outputs (list, optional): Outputs that this suffix is valid for.
        skip_outputs (list, optional): Outputs that this suffix is not
            valid for.

    """

    def __init__(self, name, value=NoDefault, prefix=NoDefault,
                 suffix=NoDefault, cond=NoDefault, noteq=NoDefault,
                 default=NoDefault, title=NoDefault, conv=NoDefault,
                 sep=NoDefault, composite=NoDefault,
                 outputs=NoDefault, skip_outputs=NoDefault):
        if prefix is NoDefault:
            prefix = '_'
        if suffix is NoDefault:
            suffix = ''
        if title is NoDefault:
            title = False
        if isinstance(outputs, str):
            outputs = [outputs]
        if isinstance(skip_outputs, str):
            skip_outputs = [skip_outputs]
        assert not (cond is not NoDefault and noteq is not NoDefault)
        self.name = name
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
        self.composite = composite
        self.composite_cls = NoDefault
        self.composite_args = NoDefault

    def is_valid(self, output):
        r"""Check if the suffix is valid for the provided output.

        Args:
            output (str): Name of output to check.

        Returns:
            bool: True if the suffix is valid, False otherwise.

        """
        if self.outputs is not NoDefault and output not in self.outputs:
            return False
        if ((self.skip_outputs is not NoDefault
             and output in self.skip_outputs)):
            return False
        return True

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
        assert isinstance(self.name, str)
        if self.composite is NoDefault:
            return [self.name]
        if self.composite_args is NoDefault:
            self.composite_cls = get_class_registry().get(
                'argument', self.composite)
            self.composite_args = self.composite_cls.argument_names(
                self.name)
        return self.composite_args

    def get_value(self, args, output, wildcards):
        r"""Get the value for an argument.

        Args:
            args (argparse.Namespace): Parsed arguments.
            output (str): Name of output to generate suffix for.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.

        Returns:
            object: Argument value.

        """
        if hasattr(args, self.name):
            value = getattr(args, self.name)
        else:
            if self.name in wildcards:
                return None
            # value = TaskBase.get_output_task(output).arg2default(
            #     self.name)
            assert hasattr(args, self.name)
        if isinstance(value, CompositeArgument):
            if wildcards:
                value = value.string_glob(wildcards)
            elif value.string is None:
                value = value.value
            else:
                value = value.string
        return value

    def _value2str(self, x):
        if isinstance(x, str):
            return x
        if ((isinstance(x, (list, np.ndarray, units.QuantityArray))
             and self.sep is not NoDefault)):
            return self.sep.join([self._value2str(xx) for xx in x])
        # Allow precision?
        # TODO: Handle arithmetic operators
        if isinstance(x, (float, units.Quantity)):
            return str(x).replace('.', 'p').replace(' ', '')
        return str(x)

    def _callable(self, x):
        return (x is not NoDefault and callable(x))

    def eval_condition(self, args, output, wildcards,
                       cond=NoDefault, value=NoDefault):
        r"""Evaluate the condition that determines if the suffix should
        be generated or default (if provided) should be used.

        Args:
            args (argparse.Namespace): Parsed arguments.
            output (str): Name of output to generate suffix for.
            wildcards (list): List of arguments that wildcards
                should be used for in the generated output file name.

        Returns:
            bool: Value of the evaluated condition. If True, the suffix
                should be generated from the argument value. If False
                and a default is provided, the default should be used.
                If False and a default is not provided, an empty suffix
                should be returned.

        """
        if cond is NoDefault:
            if value is NoDefault:
                value = self.get_value(args, output, wildcards)
            if self.noteq is not NoDefault:
                noteq = (self.noteq(args) if self._callable(self.noteq)
                         else self.noteq)
                cond = (value != noteq)
            else:
                cond = self.cond
            if cond is NoDefault:
                try:
                    cond = bool(value)
                except ValueError:
                    cond = bool(len(value))
        if self._callable(cond):
            try:
                cond = cond(args)
            except AttributeError:
                print(output, self.name)
                pdb.set_trace()
                raise
        return cond

    def __call__(self, args, output, wildcards=None):
        r"""Generate the suffix string for this argument by inspecting
        args.

        Args:
            args (argparse.Namespace): Parsed arguments.
            output (str): Name of output to generate suffix for.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.

        Returns:
            str: Generated suffix.

        """
        return self.generate(args, output, wildcards=wildcards)

    def generate(self, args, output, wildcards=None, force=False,
                 value=NoDefault):
        r"""Generate the suffix string for this argument by inspecting
        args.

        Args:
            args (argparse.Namespace): Parsed arguments.
            output (str): Name of output to generate suffix for.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.
            force (bool, optional): If True, force the condition to be
                True.
            value (object, optional): Value that should be used instead
                of the args attribute.

        Returns:
            str: Generated suffix.

        """
        if not self.is_valid(output):
            return ''
        if self.name == 'output':
            assert value is NoDefault
            value = output
            force = True
        if wildcards is None:
            wildcards = []
        if self.name in wildcards:
            return '*'
        value0 = (value if value is not NoDefault
                  else self.get_value(args, output, wildcards))
        value = self.value
        if self._callable(value):
            value = value(args)
        if value is NoDefault:
            value = value0
        cond = True if force else self.eval_condition(
            args, output, wildcards, value=value0)
        if not cond:
            if self.default is NoDefault:
                return ''
            value = self.default
        if self.conv is not NoDefault:
            value = self.conv(value)
        value = self._value2str(value)
        if ((any(x in value for x in ',:[](){};\"\'<>/+ ')
             or ((not wildcards) and '*' in value))):
            raise SuffixGenerationError(
                f'{self.name} suffix contains invalid characters: '
                f'{value}'
            )
        if self.title:
            value = value.title()
        out = f'{self.prefix}{value}{self.suffix}'
        if wildcards:
            while '**' in out:
                out = out.replace('**', '*')
        return out


class ArgumentSetSuffix(ArgumentSuffix):
    r"""Class for generating a suffix from a set of arguments.

    Args:
        names (list): Argument names.
        kwargs_set (dict, optional): Keyword arguments for the set
            itself.
        **kwargs: Additional keyword arguments are passed to
            to ArgumentSuffix constructors for each argument.
            If different values for these keywords are required for each
            argument, a dictionary should be provided.

    """

    _require_all = True

    def __init__(self, names, kwargs_set=None, **kwargs):
        if kwargs_set is None:
            kwargs_set = {}
        for k in kwargs.keys():
            assert not k.startswith('set_')
        # kwargs_set = {
        #     k: kwargs.pop(f'set_{k}') for k in [
        #         'sep', 'prefix', 'suffix', 'cond'
        #     ] if k in kwargs
        # }
        kwargs_set.setdefault('prefix', '_')
        kwargs_set.setdefault('suffix', '')
        kwargs_set.setdefault('sep', '_')
        kwargs.setdefault('prefix', '')
        kwargs.setdefault('suffix', '')
        for k in list(kwargs.keys()):
            if not isinstance(kwargs.get(k, NoDefault), dict):
                kwargs[k] = {name: kwargs.get(k, NoDefault)
                             for name in names}
        self.arguments = OrderedDict()
        for name in names:
            ikws = {k: v[name] for k, v in kwargs.items()}
            self.arguments[name] = ArgumentSuffix(name, **ikws)
        super(ArgumentSetSuffix, self).__init__(names, **kwargs_set)

    def depends(self, output):
        r"""Determine what arguments this suffix depends on for the
        provided output.

        Args:
            output (str): Name of output to check.

        Returns:
            list: Names of arguments that the suffix depends on.

        """
        out = []
        if not self.is_valid(output):
            return out
        for v in self.arguments.values():
            out += v.depends(output)
        return out

    def get_value(self, args, output, wildcards):
        r"""Get the value for an argument.

        Args:
            args (argparse.Namespace): Parsed arguments.
            output (str): Name of output to generate suffix for.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.

        Returns:
            object: Argument value.

        """
        values = OrderedDict()
        for k, v in self.arguments.items():
            values[k] = v.generate(args, output, wildcards=wildcards)
        if not self._require_all:
            return [v for v in values.values() if v]
        if not any(values.values()):
            return ''
        if any(values.values()) and not all(values.values()):
            for k, v in self.arguments.items():
                if values[k]:
                    continue
                values[k] = v.generate(args, output, wildcards=wildcards,
                                       force=True)
        return list(values.values())


class SuffixGenerator(ArgumentSetSuffix):
    r"""Class for creating a file suffix.

    Args:
        arguments (dict): Mapping between argument names and
            ArgumentSuffix construction keyword arguments for those
            arguments.
        **kwargs: Additional keyword arguments are passed to the
            ArgumentSetSuffix constructor as kwargs_set.

    """

    _require_all = False

    def __init__(self, arguments, **kwargs):
        assert 'kwargs_set' not in kwargs
        kwargs.setdefault('sep', '')
        kwargs.setdefault('prefix', '')
        kwargs.setdefault('suffix', '')
        super(SuffixGenerator, self).__init__([], kwargs_set=kwargs)
        if isinstance(arguments, list):
            arguments = OrderedDict(arguments)
        for k, v in arguments.items():
            if isinstance(v, ArgumentSuffix):
                self.arguments[k] = v
            elif isinstance(k, tuple):
                self.arguments[k] = ArgumentSetSuffix(k, **v)
            else:
                self.arguments[k] = ArgumentSuffix(k, **v)


class TaskBase(SubparserBase):
    r"""Base class for tasks.

    Args:
        args (argparse.Namespace): Parsed arguments.
        root (TaskBase, optional): Top level task.
        cached_outputs (dict, optional): Outputs that have been cached in
            memory.

    Class Attributes:
        _output_info (dict): Properties of task outputs.
        _outputs_local (list): Outputs produced by this task.
        _outputs_required (list): Required local outputs.
        _outputs_optional (list): Optional local outputs.
        _outputs_external (dict): Mapping between outputs produced by
            external tasks that are used by this task and the external
            task that produces it.
        _outputs_total (list): All outputs produced by this task or the
            external tasks that are used by this task.
        _external_tasks (dict): Mapping of external tasks used by this
            task and information about how arguments should be adopted
            by this task.
        _dont_inherit_base (bool): If True, arguments of the base class
            will not be inherited.
        _arguments (list): Class arguments.

    """

    _registry_key = 'task'
    _default = 'generate'
    _output_info = {}
    _outputs_local = []
    _outputs_required = []
    _outputs_optional = []
    _outputs_external = {}
    _outputs_total = []
    _external_tasks = {}
    _dont_inherit_base = False
    _arguments = [
        (('--verbose', ), {
            'action': 'store_true',
            'help': 'Show log messages'
        }),
        (('--debug', ), {
            'action': 'store_true',
            'help': ('Run in debug mode, setting break points for debug '
                     'messages and errors')
        }),
        (('--output-dir', ), {
            'type': str, 'default': cfg['directories']['output'],
            'help': 'Base directory where output should be stored.',
        }),
    ]

    @staticmethod
    def _on_registration(cls):
        SubparserBase._on_registration(cls)
        base = inspect.getmro(cls)[1]
        if base == SubparserBase:
            return
        cls._build_arguments(cls)

    def __init__(self, args=None, root=None, cached_outputs=None):
        if args is None:
            args = self.copy_external_args(
                args_overwrite={self._registry_key: self._name},
                set_defaults=True,
            )
        if root is None:
            root = self
        self.root = root
        self._outputs = {}
        if root is self:
            if cached_outputs is None:
                cached_outputs = {}
            self._cached_outputs = cached_outputs
        else:
            assert cached_outputs is None
            self._cached_outputs = root._cached_outputs
        self.external_tasks = {}
        super(TaskBase, self).__init__()
        self.args = args
        for cls in self._external_tasks.keys():
            if cls._name in self.external_tasks:
                continue
            else:
                self.external_tasks[cls._name] = cls(
                    self.args, root=self.root,
                )
        if self.is_root:
            args._parameter_inst_function = self.get_parameter_inst
            self.finalize()

    def finalize(self, dont_overwrite=False):
        r"""Perform steps to finalize the class for use.

        Args:
            dont_overwrite (bool, optional): If True, don't remove
                overwritten files.

        """
        self.adjust_args(self.args)
        if not dont_overwrite:
            self.overwrite_outputs()

    @property
    def is_root(self):
        r"""bool: True if this is the root task."""
        return (self.root is self)

    def get_parameter_inst(self):
        r"""Get the instance of ParametrizeCropTask used by this task.

        Returns:
             ParametrizeCropTask: Instance that parametrizes geometries.

        """
        return self.output_task('parametrize', from_root=True)

    @classmethod
    def task_hierarchy(cls):
        r"""list: Order that outputs should be initialized in."""
        out = []
        for v in cls._external_tasks:
            out += [x for x in v.task_hierarchy() if x not in out]
        out.append(cls._name)
        return out

    @classmethod
    def adjust_args(cls, args, subset=None):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            subset (list, optional): Set of arguments that should be
                adjusted. If not provided, all of the available arguments
                identified by the available options will be performed.
                    'internal': Internal arguments will be adjusted by
                        this task.
                    'external': External arguments will be adjusted by
                        external tasks.
                    'outputs': Output arguments will be adjusted.

        """
        if subset is None:
            subset = ['internal', 'external', 'outputs']
        if 'internal' in subset:
            cls.adjust_args_internal(args)
        if 'external' in subset:
            # Omit outputs so that all arguments are initialized before
            # OutputArgument instances are created
            cls.adjust_args_external(
                args, subset=['internal', 'external'],
            )
        if 'outputs' in subset:
            cls.adjust_args_output(args)

    @classmethod
    def adjust_args_internal(cls, args):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        valid_arguments = cls.argument_dict()
        for k, v in valid_arguments.items():
            if not hasattr(args, k):
                setattr(args, k, cls.arg2default(k, info=v[1]))
        order = list(cls._argument_conversions.keys())
        for k in order:
            if k == 'output':
                continue
            if k in cls._composite_arguments:
                for kk in cls._argument_conversions[k]:
                    if kk not in valid_arguments:
                        continue
                    cls._convert_composite(args, kk, base=k)
            else:
                method = getattr(cls, f'_convert_{k}')
                for kk in cls._argument_conversions[k]:
                    if kk not in valid_arguments:
                        continue
                    method(args, kk, getattr(args, kk, None))

    @classmethod
    def adjust_args_external(cls, args, **kwargs):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments for external tasks.

        Args:
            args (argparse.Namespace): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to
                adjust_args for each external task.

        """
        for ext in cls._external_tasks.keys():
            ext.adjust_args(args, **kwargs)

    @classmethod
    def adjust_args_output(cls, args):
        r"""Create the output arguments (assumes that adjust_args has
        already been called to initialize the other arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        for ext in cls._external_tasks.keys():
            ext.adjust_args_output(args)
        for k in cls._argument_conversions['output']:
            cls._convert_composite(args, k, base='output')

    @classmethod
    def complete_external_args(cls, args):
        r"""Add missing arguments to the provided argument set.

        Args:
            args (argparse.Namespace): Parsed arguments to complete.

        """
        setattr(args, cls._registry_key, cls._name)
        for k, props in cls.argument_dict(
                include_subparser='explicit').items():
            if hasattr(args, k):
                continue
            setattr(args, k, cls.arg2default(k, info=props[1]))
        cls.adjust_args(args)

    @classmethod
    def copy_external_args(cls, args=None, args_overwrite=None,
                           args_external=None, set_defaults=False,
                           initialize=False, return_dict=False,
                           verbose=False):
        r"""Extract arguments for this task from a set for another task.

        Args:
            args (argparse.Namespace, optional): Parsed arguments to
                copy.
            args_overwrite (dict, optional): Arguments to overwrite.
            args_external (list, optional): External arguments that should
                be preserved.
            set_defaults (bool, optional): If True, don't set missing
                arguments to defaults.
            initialize (bool, optional): If True, initialize arguments
                by applying adjust_args (implies set_defaults == True).
            return_dict (bool, optional): Return a dictionary instead
                of argparse.Namespace.
            verbose (bool, optional): If True, write copied arguments
                to a log message.

        Returns:
            argparse.Namespace: Arguments.

        """
        if initialize:
            set_defaults = True
        if args is None:
            args = argparse.Namespace()
        if args_overwrite is None:
            args_overwrite = {}
        if args_external is None:
            args_external = []
        args_overwrite.setdefault(cls._registry_key, cls._name)
        args_changing = list(args_overwrite.keys())
        if args_changing:
            for k, kargs in cls._composite_arguments.items():
                for kk in kargs:
                    if kk not in args_changing:
                        continue
                    v = get_class_registry().get('argument', k)
                    args_changing += [
                        x for x in v.argument_names(kk, no_universal=True)
                        if x not in args_changing
                    ]
        if 'id' not in args_changing:
            args_external += ['_parameter_inst_function']
        out = {}
        for k, props in cls.argument_dict(
                include_subparser='explicit').items():
            if k in args_overwrite:
                out[k] = args_overwrite[k]
            elif k in out:
                continue
            elif hasattr(args, k):
                out[k] = getattr(args, k)
            elif set_defaults:
                out[k] = cls.arg2default(k, info=props[1])
            else:
                out[k] = None
            v = out[k]
            if isinstance(v, CompositeArgument):
                for kk, vv in v.raw_args(k).items():
                    if kk in args_overwrite:
                        continue
                    out[kk] = vv
                continue
        missing = []
        for k in args_overwrite.keys():
            if k not in out:
                missing.append(k)
        if missing:
            raise Exception(f'Members of args_overwrite were unused: '
                            f'{missing}')
        for k in args_external:
            assert k not in out
            if hasattr(args, k):
                out[k] = getattr(args, k)
        if verbose:
            cls.log_class(pprint.pformat(out))
        if return_dict:
            assert not initialize
            return out
        kws = out
        out = argparse.Namespace()
        for k, v in kws.items():
            setattr(out, k, v)
        if initialize:
            cls.adjust_args(out)
        return out

    @classmethod
    def from_kwargs(cls, kws, **kwargs):
        r"""Create an instance from the provided arguments.

        Args:
            kws (dict): Keyword arguments that should be parsed into
                arguments via the parse function.
            **kwargs: Additional keyword arguments are passed to the
                class constructor.

        """
        kws[cls._registry_key] = cls._name
        args = cls.copy_external_args(args_overwrite=kws,
                                      set_defaults=True)
        return cls(args, **kwargs)

    @classmethod
    def from_external_args(cls, args, args_overwrite=None,
                           args_external=None, copy_outputs_from=None,
                           verbose=False, **kwargs):
        r"""Create an instance from a set of external arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            args_ovewrite (dict, optional): Argument values to set for
                the run after copying the current argument namespace.
            args_external (list, optional): External arguments that should
                be preserved.
            copy_outputs_from (TaskBase, optional): Existing instance
                that matching outputs should be copied from.
            verbose (bool, optional): If True, write copied arguments
                to a log message.
            **kwargs: Additional keyword arguments are passed to the
                class constructor.

        Returns:
            TaskBase: New task instance.

        """
        args = cls.copy_external_args(
            args, args_overwrite=args_overwrite,
            args_external=args_external, set_defaults=True,
            verbose=verbose,
        )
        out = cls(args, **kwargs)
        if copy_outputs_from is not None:
            out.copy_matching_outputs(copy_outputs_from)
        return out

    # Methods for building arguments
    @staticmethod
    def _build_arguments(cls):
        for k in ['_subparser_arguments', '_subparser_modifications',
                  '_arguments', '_argument_conversions',
                  '_output_info', '_outputs_required', '_outputs_local',
                  '_composite_arguments']:
            setattr(cls, k, copy.deepcopy(getattr(cls, k)))
        if not cls._dont_inherit_base:
            base = inspect.getmro(cls)[1]
            TaskBase._copy_external_arguments(cls, base)
        cls._dont_inherit_base = False
        if cls._name is not None and cls._name not in cls._outputs_required:
            cls._outputs_required.insert(0, cls._name)
        cls._outputs_required = [
            k for k, v in cls._output_info.items()
            if not v.get('optional', False)
        ]
        cls._outputs_optional = [
            k for k, v in cls._output_info.items()
            if v.get('optional', False)
        ]
        cls._outputs_local = cls._outputs_required + cls._outputs_optional
        cls._outputs_external = {}
        for ext, props in cls._external_tasks.items():
            TaskBase._copy_external_arguments(cls, ext, **props)
        cls._outputs_total = (
            cls._outputs_local + list(cls._outputs_external.keys())
        )
        for k, v in cls._output_info.items():
            if k in cls._outputs_external:
                continue
            if 'base' in v and v['base'] not in v.get('upstream', []):
                v.setdefault('upstream', [])
                v['upstream'].append(v['base'])
            for x in v.get('upstream', []):
                xsrc = cls
                while x not in xsrc._outputs_local:
                    xsrc = xsrc._outputs_external[x]
                xsrc._output_info[x].setdefault('downstream', {})
                xsrc._output_info[x]['downstream'].setdefault(cls, [])
                if k not in xsrc._output_info[x]['downstream'][cls]:
                    xsrc._output_info[x]['downstream'][cls].append(k)
            cls._composite_arguments.setdefault('output', {})
            koutput = f'output_{k}'
            if koutput not in cls._composite_arguments['output']:
                cls._composite_arguments['output'][koutput] = v
        CompositeArgument.add_arguments(cls)

    @classmethod
    def _get_subparser_arguments(cls, k, use_flags=False):
        assert k in cls._subparser_arguments
        kargs = cls._subparser_arguments[k]
        if isinstance(kargs, list):
            if use_flags:
                kargs = ['--' + x.replace('_', '-') for x in kargs]
            return kargs
        if kargs is False:
            return []
        assert kargs is True
        kargs = []
        for v in get_class_registry().values(k):
            kargs += [
                x for x in v.argument_names(use_flags=use_flags)
                if x not in kargs
            ]
        return kargs

    @staticmethod
    def _copy_external_arguments(dst, src, include=None, exclude=None,
                                 modifications=None, optional=False):
        arguments = dst.argument_dict()
        if modifications is None:
            modifications = {}
        modifications = copy.deepcopy(modifications)

        def check(k):
            if exclude is not None and k in exclude:
                return False
            if include is not None and k not in include:
                return False
            return True

        if include is not None:
            include += [f'output_{k}' for k in src._outputs_local]
            if optional:
                for k in src._outputs_local:
                    modifications.setdefault(f'output_{k}', {})
                    modifications[f'output_{k}'].setdefault(
                        'default', False)
        for argT, v in src._composite_arguments.items():
            arg_class = get_class_registry().get('argument', argT)
            for k in v.keys():
                prefix = arg_class.get_prefix(k)
                suffix = arg_class.get_suffix(k)
                added_args = arg_class.argument_names(
                    prefix=prefix, suffix=suffix)
                if exclude is not None and k in exclude:
                    exclude = exclude + [
                        kk for kk in added_args if kk not in exclude]
                if include is not None and k in include:
                    include = include + [
                        kk for kk in added_args if kk not in include]
        for k in src._subparser_arguments.keys():
            subparser_args = None
            if modifications:
                subparser_args = src._get_subparser_arguments(k)
                for kmod in modifications.keys():
                    if kmod in subparser_args:
                        dst._subparser_modifications.setdefault(k, {})
                        dst._subparser_modifications[k].setdefault(
                            kmod, modifications[kmod])
            if (not check(k)) or k in dst._subparser_arguments:
                continue
            if include is None and exclude is None:
                dst._subparser_arguments[k] = copy.deepcopy(
                    src._subparser_arguments[k])
                continue
            if subparser_args is None:
                subparser_args = src._get_subparser_arguments(k)
            dst._subparser_arguments[k] = [
                x for x in subparser_args if check(x)
            ]
        for k, v in src._subparser_modifications.items():
            if (not check(k)) or k in dst._subparser_modifications:
                continue
            dst._subparser_modifications[k] = copy.deepcopy(
                src._subparser_modifications[k])
        for k, v in src.argument_dict().items():
            if v[1].get('subparser_specific_dest', False):
                k = k.rsplit(f'_{src._name}', 1)[0] + f'_{dst._name}'
            if not check(k):
                continue
            if k not in arguments:
                arguments[k] = v
            for argT, v in src._composite_arguments.items():
                if k in v:
                    if k not in dst._composite_arguments.get(argT, {}):
                        dst._composite_arguments.setdefault(argT, {})
                        dst._composite_arguments[argT][k] = (
                            copy.deepcopy(v[k])
                        )
                        if optional and argT == 'output':
                            dst._composite_arguments[argT][k].setdefault(
                                'defaults', {})
                            dst._composite_arguments[argT][k][
                                'defaults']['output'] = False
                    break
        if modifications:
            for k, v in modifications.items():
                if k not in arguments:
                    continue
                arguments[k] = SubparserBase._modify_argument(
                    arguments[k], **v)
        dst._arguments = list(arguments.values())
        dst._outputs_external.update(src._outputs_external)
        for k in src._outputs_total:  # TODO: Verify this works
            dst._outputs_external[k] = src
        for k, v in src._argument_conversions.items():
            dst._argument_conversions.setdefault(k, [])
            dst._argument_conversions[k] += [
                vv for vv in v
                if vv not in dst._argument_conversions[k]
                and check(vv)
            ]
            if not dst._argument_conversions[k]:
                dst._argument_conversions.pop(k)
                continue
            if ((k in dst._composite_arguments
                 or hasattr(dst, f'_convert_{k}'))):
                continue
            if not hasattr(dst, f'_convert_{k}'):
                setattr(dst, f'_convert_{k}',
                        getattr(src, f'_convert_{k}'))

    # Methods for managing task I/O
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
        ext = os.path.splitext(fname)[-1]
        if ext == '.json':
            with open(fname, 'r') as fd:
                return rapidjson.load(fd)
        elif ext == '.csv':
            return utils.read_csv(fname, verbose=args.verbose)
        elif ext in ['.obj', '.ply']:
            return utils.read_3D(
                fname, file_format=getattr(args, 'mesh_format', None),
                verbose=args.verbose,
                # include_units=getattr(args, 'include_units', False),
            )
        elif ext == '.png':
            return utils.read_png(fname, verbose=args.verbose)
        elif ext in ['.lpy']:
            with open(fname, 'r') as fd:
                return fd.read()
        raise NotImplementedError(f'{name}: {fname}')

    def read_output(self, name=None):
        r"""Load an output file produced by this task.

        Args:
            name (str, optional): Name of the output to read.

        Returns:
            object: Contents of the output file.

        """
        def _read_output(task, name):
            fname = task.output_file(name, return_disabled=True)
            if not os.path.isfile(fname):
                raise RuntimeError(f'\"{name}\" output file does not '
                                   f'exist: {fname}')
            return task._read_output(task.args, name, fname)

        return self._call_output_task(self, _read_output, name)

    @classmethod
    def _write_output(cls, args, name, fname, output):
        r"""Write to an output file.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Name of the output to write.
            fname (str): Path of the file that should be written to.
            output (object): Output object to write to file.

        """
        ext = os.path.splitext(fname)[-1]
        if ext == '.json':
            assert output
            with open(fname, 'w') as fd:
                rapidjson.dump(output, fd,
                               write_mode=rapidjson.WM_PRETTY)
            return
        elif ext == '.csv':
            utils.write_csv(output, fname, verbose=args.verbose)
            return
        elif ext in ['.obj', '.ply']:
            utils.write_3D(output, fname,
                           file_format=args.mesh_format,
                           verbose=args.verbose)
            return
        elif ext == '.png':
            from matplotlib.figure import Figure
            if isinstance(output, Figure):
                output.savefig(fname, dpi=300)
                return
            utils.write_png(output, fname, verbose=args.verbose)
            return
        elif ext in ['.lpy']:
            with open(fname, 'w') as fd:
                fd.write(output)
            return
        raise NotImplementedError(f'{name}: {fname}')

    def write_output(self, name, output, overwrite=False):
        r"""Write to an output file.

        Args:
            name (str): Name of the output to write.
            output (object): Output object to write to file.
            overwrite (bool, optional): If True, overwrite existing
                output.

        """

        def _write_output(task, name):
            if not task.output_enabled(name, for_write=True):
                raise RuntimeError(f'Write to disk not enabled for '
                                   f'\"{name}\" output.')
            fname = task.output_file(name)
            assert isinstance(fname, str)
            if (not overwrite) and os.path.isfile(fname):
                return
            fdir = os.path.dirname(fname)
            task.log(f'Writing \"{name}\" output: {fname}',
                     force=True)
            if not os.path.isdir(self.args.output_dir):
                os.mkdir(self.args.output_dir)
            if not os.path.isdir(fdir):
                os.mkdir(fdir)
            task._write_output(task.args, name, fname, output)

        return self._call_output_task(self, _write_output, name)

    def output_task(self, name, default=NoDefault, from_root=False):
        r"""Get the task instance responsible for producing an output.

        Args:
            name (str): Name of output to get task for.
            default (object, optional): Value to return if the task
                instance cannot be located.
            from_root (bool, optional): If True, start from the root
                task.

        Returns:
            TaskBase: Task that produces the output.

        """
        if from_root:
            return self.root._get_output_task(self.root, name,
                                              default=default)
        return self._get_output_task(self, name, default=default)

    @classmethod
    def get_output_task(cls, name, default=NoDefault):
        r"""Determine which task class produces the named output.

        Args:
            name (str): Output name.
            default (object, optional): Value to return if the task
                class cannot be located.

        Returns:
            type: Task class.

        """
        for v in get_class_registry().values('task'):
            if name in v._outputs_local:
                return v
        if default is not NoDefault:
            return default
        raise KeyError(name)

    @staticmethod
    def _get_output_task(cls, name=None, default=NoDefault,
                         initialize_missing=False):
        r"""Get the task class responsible for producing an output.

        Args:
            name (str, list, optional): Name(s) of one or more outputs
                to get task(s) for.
            default (object, optional): Value to return if the task
                instance cannot be located.
            initialize_missing (bool, optional): If the task instance
                does not exist, create it.

        Returns:
            type: Task class.

        """
        if isinstance(name, list):
            return {k: cls._get_output_task(cls, k) for k in name}
        if name is None:
            name = cls._name
        if name in cls._outputs_local:
            return cls
        out = cls._outputs_external[name]
        if not isinstance(cls, type):
            if out._name not in cls.external_tasks:
                if initialize_missing:
                    cls.external_tasks[out._name] = out(cls.args)
                elif default is not NoDefault:
                    return default
            out = cls.external_tasks[out._name]
        if name not in out._outputs_local:
            return cls._get_output_task(
                out, name, default=default,
                initialize_missing=initialize_missing,
            )
        return out

    @staticmethod
    def _call_output_task(cls, method, name, args=None):
        if isinstance(name, list):
            return {k: cls._call_output_task(cls, method, k, args)
                    for k in name}
        if name is None:
            name = cls._name
        task = cls._get_output_task(cls, name)
        return method(task, name)

    def enabled_outputs(self, for_write=False):
        r"""Get the set of outputs enabled by the provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            for_write (bool, optional): If True, only return outputs
                for which write is enabled.

        Returns:
            list: Names of enabled outputs.

        """
        return [k for k in self._outputs_local
                if self.output_enabled(k, for_write=for_write)]

    def output_exists(self, name=None):
        r"""Check if a task output exists.

        Args:
            name (str, list, optional): Name(s) of one or more outputs
                to check for.

        Returns:
            bool: True if the output file exists.

        """
        if isinstance(name, list):
            return all(self.output_exists(name=x) for x in name)
        if name is None:
            name = self._name
        task = self._get_output_task(self, name)
        output = getattr(task.args, f'output_{name}')
        if (not output.overwrite) and name in task._outputs:
            return True
        return ((not output.overwrite) and output.exists)

    def output_missing(self, name=None):
        r"""Check if a task output is enabled, but does not exist.

        Args:
            name (str, list, optional): Name(s) of one or more outputs
                to check for.

        Returns:
            bool: True if the output file exists.

        """
        if isinstance(name, list):
            return any(self.output_missing(name=x) for x in name)
        if name is None:
            name = self._name
        task = self._get_output_task(self, name)
        output = task.output_file(name)
        if not output:
            return False
        return (not task.output_exists(name))

    @classmethod
    def _output_enabled(cls, args, name=None, for_write=False):
        r"""Check if an output is enabled by the current arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, list, optional): Name(s) of one or more outputs
                to check.
            for_write (bool, optional): If True, only return True if
                output is enabled and write is not disabled.

        Returns:
            bool: True if the output is enabled, False otherwise.

        """

        def _output_enabled(task, name):
            output = getattr(args, f'output_{name}')
            if for_write and (output.dont_write or output.is_unmerged):
                return False
            return output.enabled

        return cls._call_output_task(cls, _output_enabled, name)

    def output_enabled(self, name=None, for_write=False):
        r"""Check if an output is enabled by the current arguments.

        Args:
            name (str, list, optional): Name(s) of one or more outputs
                to check.
            for_write (bool, optional): If True, only return True if
                output is enabled and write is not disabled.

        Returns:
            bool: True if the output is enabled, False otherwise.

        """

        def _output_enabled(task, name):
            return task._output_enabled(task.args, name,
                                        for_write=for_write)

        return self._call_output_task(self, _output_enabled, name)

    @classmethod
    def _output_names(cls, args,
                      exclude_local=False, include_external=False,
                      exclude_optional=False, exclude_required=False,
                      include_disabled=False, for_write=False):
        r"""Get the list of outputs that will be generated.

        Args:
            args (argparse.Namespace): Parsed arguments.
            exclude_local (bool, optional): If True, don't include
                local outputs.
            include_external (bool, optional): If True, include outputs
                produced by external tasks.
            exclude_optional (bool, optional): If True, don't include
                optional outputs.
            exclude_required (bool, optional): If True, don't include
                required outputs.
            include_disabled (bool, optional): If True, include disabled
                outputs.
            for_write (bool, optional): If True, only include outputs
                that will be written to disk

        Returns:
            list: Output names.

        """
        out = []
        if not exclude_local:
            if not exclude_optional:
                out += cls._outputs_optional
            if not exclude_required:
                out += cls._outputs_required
            if not include_disabled:
                out = [k for k in out if
                       cls._output_enabled(args, k, for_write=for_write)]
        if include_external:
            for k, v in cls._external_tasks.items():
                if exclude_optional and v.get('optional', False):
                    continue
                out += [
                    x for x in k._output_names(
                        args, include_external=include_external,
                        exclude_optional=exclude_optional,
                        exclude_required=exclude_required,
                        include_disabled=include_disabled,
                        for_write=for_write)
                    if x not in out
                ]
        return out

    def output_names(self, **kwargs):
        r"""Get the list of outputs that will be generated.

        Args:
            **kwargs: Keyword arguments are passed to _output_names.

        Returns:
            list: Output names.

        """
        return self._output_names(self.args, **kwargs)

    @classmethod
    def _make_suffix_set(cls, args, names, set_sep=NoDefault,
                         set_prefix='_', set_suffix='',
                         wildcards=None, skip=None, **kwargs):
        r"""Create a suffix by combining multiple arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            names (list): Argument names.
            set_sep (str, optional): Separator to use between arguments.
            set_prefix (str, optional): Prefix to use before set suffix.
            set_suffix (str, optional): Suffix to use after set suffix.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.
            skip (list, optional): Arguments to skip.
            **kwargs: Additional keyword arguments are passed to
                _make_suffix. If different values for these keywords are
                required for each argument, a dictionary should be
                provided.

        Returns:
            str: Generated suffix.

        """
        kwargs.setdefault('prefix', '')
        kwargs.setdefault('suffix', '')
        for k in list(kwargs.keys()):
            if not isinstance(kwargs.get(k, NoDefault), dict):
                kwargs[k] = {name: kwargs.get(k, NoDefault)
                             for name in names}
        values = []
        for name in names:
            ikws = {k: v[name] for k, v in kwargs.items()}
            ivalue = cls._make_suffix(
                args, name, wildcards=wildcards, skip=skip, **ikws
            )
            if ivalue == '*':
                return '*'
            values.append(ivalue)
        if not any(values):
            return ''
        if any(values) and not all(values):
            for i in range(len(values)):
                if values[i]:
                    continue
                ikws = {k: v[name] for k, v in kwargs.items()}
                ikws.update(cond=True, noteq=NoDefault)
                values[i] = cls._make_suffix(
                    args, name, wildcards=wildcards, skip=skip, **ikws
                )
        if set_sep is NoDefault:
            set_sep = '_'
        return set_prefix + set_sep.join(values) + set_suffix

    @classmethod
    def _make_suffix(cls, args, name, value=NoDefault, prefix=NoDefault,
                     suffix=NoDefault, cond=NoDefault, noteq=NoDefault,
                     default=NoDefault, title=NoDefault, conv=NoDefault,
                     sep=NoDefault, wildcards=None, skip=None):
        r"""Create a suffix associated with an argument.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, list): Argument name(s).
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
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.
            skip (list, optional): Arguments to skip.

        Returns:
            str: Generated suffix.

        """
        assert name is not None
        if skip is not None and name in skip:
            return ''
        if wildcards and name in wildcards:
            return '*'
        if prefix is NoDefault:
            prefix = '_'
        if suffix is NoDefault:
            suffix = ''
        if title is NoDefault:
            title = False
        if not hasattr(args, name):
            if wildcards and name in wildcards:
                value0 = None
            else:
                value0 = cls.arg2default(name)
        else:
            value0 = getattr(args, name)
        if isinstance(value0, CompositeArgument):
            if wildcards:
                value0 = value0.string_glob(wildcards)
            elif value0.string is None:
                value0 = value0.value
            else:
                value0 = value0.string
        if value is NoDefault:
            value = value0
        if noteq is not NoDefault:
            assert cond is NoDefault
            cond = (value0 != noteq)
        elif cond is NoDefault:
            assert noteq is NoDefault
            try:
                cond = bool(value0)
            except ValueError:
                cond = bool(len(value0))
        if not cond:
            if default is NoDefault:
                return ''
            value = default
        if conv is not NoDefault:
            value = conv(value)

        def float2str(x):
            # Allow precision?
            # TODO: Handle arithmetic operators
            if isinstance(x, (float, units.Quantity)):
                return str(x).replace('.', 'p').replace(' ', '')
            return str(x)

        if ((isinstance(value, (list, np.ndarray, units.QuantityArray))
             and sep is not NoDefault)):
            value = sep.join([float2str(x) for x in value])
        if not isinstance(value, str):
            value = float2str(value)
        if ((any(x in value for x in ',:[](){};\"\'<>/+ ')
             or ((not wildcards) and '*' in value))):
            raise SuffixGenerationError(
                f'{name} contains invalid characters: {value}'
            )
        if title:
            value = str(value).title()
        return f'{prefix}{value}{suffix}'

    @classmethod
    def _output_depends(cls, args, name, variables):
        r"""Determine if an output is dependent on a variable.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str): Name of output to check dependency of.
            variables (list): List of arguments to check if the affect
                the named output.

        Returns:
            bool: True if the output depends on the named variables.

        """
        output = cls._output_argument(args, name)
        return output.depends(variables)

    def output_depends(self, name, variables):
        r"""Determine if an output is dependent on a variable.

        Args:
            name (str): Name of output to check dependency of.
            variables (list): List of arguments to check if the affect
                the named output.

        Returns:
            bool: True if the output depends on the named variables.

        """
        return self._output_depends(self.args, name, variables)

    @classmethod
    def _output_file(cls, args, name=None, return_disabled=False,
                     regenerate=False, wildcards=None, copy_args=False):
        r"""Get the filename for an output.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, list, optional): Name(s) of one or more outputs
                to get the files for.
            return_disabled (bool, optional): If True, return the
                generated filename even if the output is disabled.
            regenerate (bool, optional): If True, regenerate all output
                file names.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.
            copy_args (bool, optional): If True, the args should be
                copied and initialized.

        Returns:
            str: Name of output file. If True, this indicates the output
                should be generated, but not written. If False, the output
                should not be generated.

        """

        def _output_file(task, name):
            output = task._output_argument(args, name,
                                           copy_args=copy_args)
            # if ((copy_args
            #      or getattr(args, f'output_{name}', None) is None)):
            #     args_local = task.copy_external_args(
            #         args, initialize=True,
            #     )
            # else:
            #     args_local = args
            # output = getattr(args_local, f'output_{name}')
            if not output.generated:
                cls.log_class(f'Filename for \"{name}\" output '
                              f'was not generated: {output.path}')
                assert output.generated
                return output.path
            if output.path and not (regenerate or wildcards):
                return output.path
            if (not return_disabled) and (not output.enabled):
                return False
            # return output.generate(args_local, wildcards=wildcards)
            return output.generate(args, wildcards=wildcards)

        return cls._call_output_task(cls, _output_file, name)

    def output_file(self, name=None, return_disabled=False,
                    regenerate=False, wildcards=None, copy_args=False):
        r"""Get the filename for an output.

        Args:
            name (str, list, optional): Name(s) of one or more outputs
                to get the files for.
            return_disabled (bool, optional): If True, return the
                generated filename even if the output is disabled.
            regenerate (bool, optional): If True, regenerate all output
                file names.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.
            copy_args (bool, optional): If True, the args should be
                copied and initialized.

        Returns:
            str: Name of output file. If True, this indicates the output
                should be generated, but not written. If False, the output
                should not be generated.

        """
        def _output_file(task, name):
            return task._output_file(
                task.args, name, return_disabled=return_disabled,
                wildcards=wildcards, regenerate=regenerate,
                copy_args=copy_args,
            )

        return self._call_output_task(self, _output_file, name)

    @classmethod
    def _output_argument(cls, args, name=None, copy_args=False):
        r"""Get the OutputArgument instance for an output.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, optional): Name of the output to return the
                argument instance for.
            copy_args (bool, optional): If True, the args should be
                copied and initialized.

        Returns:
            OutputArgument: Argument instance controling output.

        """
        if name is None:
            name = cls._name
        if getattr(args, f'output_{name}', None) is None:
            task = cls._get_output_task(cls, name)
            args = task.copy_external_args(args, initialize=True)
            getattr(args, f'output_{name}')._copied_args = args
        # if copy_args or getattr(args, f'output_{name}', None) is None:
        #     task = cls._get_output_task(cls, name)
        #     args = task.copy_external_args(args, initialize=True)
        #     args._copied_args = args
        return getattr(args, f'output_{name}')

    def output_argument(self, name=None):
        r"""Get the OutputArgument instance for an output.

        Args:
            name (str, optional): Name of the output to return the
                argument instance for.

        Returns:
            OutputArgument: Argument instance controling output.

        """
        return self._output_argument(self.args, name=name)

    @classmethod
    def _remove_output(cls, args, name=None, wildcards=None,
                       copy_args=False, force=False,
                       skip_test_output=False,
                       skip_downstream=False):
        r"""Remove existing output file.

        Args:
            args (argparse.Namespace): Parsed arguments.
            name (str, list, optional): Name(s) of one or more outputs
                to remove.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file name.
            copy_args (bool, optional): If True, the args should be
                copied and initialized.
            force (bool, optional): If True, remove without asking for
                user input.
            skip_test_output (bool, optional): If True, don't include
                both test output and generated file name.
            skip_downstream (bool, optional): If True, downstream
                outputs will not be removed.

        """
        assert not copy_args

        def _remove_output(task, name):
            output = task._output_argument(args, name)
            output.remove(args=args, wildcards=wildcards,
                          force=force, skip_test_output=skip_test_output)
            if not skip_downstream:
                output.remove_downstream(
                    args, removed=[output.output_name],
                    wildcards=wildcards, force=force,
                    skip_test_output=skip_test_output,
                )
        cls._call_output_task(cls, _remove_output, name)

    def remove_output(self, name=None, remove_local=False, **kwargs):
        r"""Remove existing output file.

        Args:
            name (str, list, optional): Name(s) of one or more outputs
                to remove.
            remove_local (bool, optional): If True, remove any in-memory
                output.
            **kwargs: Additional keyword arguments are passed to
                _remove_output.

        """
        def _remove_output(task, name):
            if remove_local:
                task._outputs.pop(name, None)
            return task._remove_output(task.args, name=name, **kwargs)

        self._call_output_task(self, _remove_output, name)

    def cache_output(self, name=None):
        r"""Cache output on the root task for later use.

        Args:
            name (str, list, optional): Name(s) of one or more outputs
                to remove.

        """
        def _cache_output(task, name):
            fname = task.output_file(name, return_disabled=True)
            task._cached_outputs[fname] = task.get_output(name)

        self._call_output_task(self, _cache_output, name)

    def overwrite_outputs(self, downstream=None, wildcards=None):
        r"""Remove existing files that should be overwritten.

        Args:
            downstream (list, optional): Existing list that output names
                should be added to that downstream files need to be
                removed for.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file names.

        """
        skip_downstream = (downstream is not None)
        if downstream is None:
            downstream = []
        for name in self._outputs_local:
            if self.overwrite_output(name, wildcards=wildcards):
                downstream.append(name)
        for task in self.external_tasks.values():
            task.overwrite_outputs(downstream=downstream,
                                   wildcards=wildcards)
        if (not skip_downstream) and downstream:
            removed = []
            for k in downstream:
                koutput = self.output_argument(k)
                koutput.remove_downstream(
                    self.args, removed=removed, wildcards=wildcards,
                )
        if self.is_root:
            for k in list(self._cached_outputs.keys()):
                if not os.path.isfile(k):
                    self._cached_outputs.pop(k)

    def overwrite_output(self, name=None, wildcards=None):
        r"""Prepare an output for a run. If overwrite specified, any
        existing data/files for the output will be removed. If the
        output file is not fully specified on the args, it will be set.

        Args:
            name (str, list, optional): Name(s) of one or more outputs
                to handle overwrite for.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file names.

        Returns:
            bool: True if downstream outputs should be removed.

        """
        if name is None:
            name = self._name
        task = self._get_output_task(self, name)
        output = task.output_argument(name)
        # fname = output.path
        remove_downstream = False
        if output.overwrite:
            task._outputs.pop(name, None)
            output.remove(task.args, wildcards=wildcards)
            remove_downstream = bool(output.downstream)
        # elif fname and not (name in task._outputs or output.exists
        #                     or output.dont_write
        #                     or output.is_unmerged
        #                     or len(output.downstream) == 0):
        #     self.log(f'Output \"{name}\" does not exist, removing '
        #              f'downstream files ({fname}) '
        #              f'isfile = {os.path.isfile(fname)}', force=True)
        #     pdb.set_trace()
        #     remove_downstream = bool(output.downstream)
        output.clear_overwrite(task.args)
        return remove_downstream

    # Methods for executing a run
    def get_output(self, name=None):
        r"""Get a task output, generating or loading it if necessary.

        Args:
            name (str, list, optional): Name(s) of one or more outputs
                to return.

        Returns:
            object: Output.

        """
        def _get_output(task, name):
            if name in task._outputs:
                return task._outputs[name]
            if task.output_exists(name=name):
                fname = task.output_file(name)
                if fname in task._cached_outputs:
                    self.log(f'Using cached \"{name}\" output: {fname}',
                             force=True)
                    task._outputs[name] = task._cached_outputs[fname]
                else:
                    self.log(f'Loading existing \"{name}\" output: '
                             f'{fname}',
                             force=True)
                    task._outputs[name] = task._read_output(
                        task.args, name, fname)
                return task._outputs[name]
            self.log(f'Generating \"{name}\"', force=True)
            task.generate_output(name)
            assert name in task._outputs
            if task.output_enabled(name, for_write=True):
                task.write_output(name, task._outputs[name])
            return task._outputs[name]

        return self._call_output_task(self, _get_output, name)

    def set_output(self, name, output, overwrite=False):
        r"""Set an output value for the task.

        Args:
            name (str): Name of the output to set.
            output (object): Output instance.
            overwrite (bool, optional): If True, overwrite existing
                output.

        """
        task = self._get_output_task(self, name)
        if not overwrite:
            assert name not in task._outputs
        task._outputs[name] = output
        if task.output_enabled(name, for_write=True):
            task.write_output(name, output, overwrite=overwrite)

    def generate_output(self, name):
        r"""Generate the specified output value.

        Args:
            name (str): Name of the output to generate.

        Returns:
            object: Generated output.

        """
        output = getattr(self.args, f'output_{name}')
        if output.is_merged or output.is_unmerged:
            assert self.args.output_generate.generated
            over = {'id': self.output_task('generate').all_ids}
            merge_all = output.is_merged
            merge_all_output = output.merge_all_output
            if merge_all_output is None:
                merge_all_output = name
            out = OrderedDict()
            i = 0
            for x in self.run_series(over=over):
                if merge_all:
                    out[over['id'][i]] = x.get_output(merge_all_output)
                i += 1
            if merge_all:
                out = self._merge_output(name, out)
            else:
                out = None
        else:
            out = self._generate_output(name)
        self.set_output(name, out)
        return out

    def _generate_output(self, name):
        r"""Generate the specified output value.

        Args:
            name (str): Name of the output to generate.

        Returns:
            object: Generated output.

        """
        raise NotImplementedError

    def _merge_output(self, name, output):
        r"""Merge the output for multiple IDs.

        Args:
            name (str): Name of the output to generate.
            output (dict): Mapping from ID to the output for each ID.

        Returns:
            object: Generated output.

        """
        raise NotImplementedError

    def run(self, output_name=None):
        r"""Run the process associated with this subparser.

        Args:
            output_name (str, optional): Name of output that should be
                returned. Defaults to the output with the same name as
                the task.

        Returns:
            object: Output named by output_name.

        """
        output_names = self.enabled_outputs()
        if output_name is None:
            output_name = self._name
        assert output_name in ['instance'] + self._outputs_local
        for k in output_names:
            assert getattr(self.args, f'overwrite_{k}') is False
            if not self.output_exists(k):
                self.get_output(k)
                assert self.output_exists(k)
        if output_name == 'instance':
            return self
        return self.get_output(output_name)

    def run_iteration(self, cls=None, **kwargs):
        r"""Run an iteration, regenerating output file names.

        Args:
            cls (type, optional): Task class that should be run in
                iteration. Defaults to the type of the current task.
            **kwargs: Additional keyword arguments are passed to
                run_iteration_class.

        Returns:
            object: Result of the run.

        """
        if cls is None:
            cls = type(self)
        kwargs.setdefault('cached_outputs', self._cached_outputs)
        return cls.run_iteration_class(self.args, **kwargs)

    @classmethod
    def run_iteration_class(cls, args, **kwargs):
        r"""Run an iteration, regenerating output file names.

        Args:
            args (argparse.Namespace): Parsed arguments.
            **kwargs: Additional keyword arguments are passed to
                run_class.

        Returns:
            object: Result of the run (defaults to the task instance).

        """
        kwargs.setdefault('output_name', 'instance')
        while True:
            try:
                return cls.run_class(args, **kwargs)
            except RepeatIterationError as e:
                kwargs['args_overwrite'] = dict(
                    kwargs.get('args_overwrite', {}),
                    **e.args_overwrite)

    @classmethod
    def run_class(cls, args, output_name=None, args_preserve=None,
                  cache_outputs=None, **kwargs):
        r"""Run the process associated with this subparser.

        Args:
            args (argparse.Namespace): Parsed arguments.
            output_name (str, optional): Name of output that should be
                returned. Defaults to the output with the same name as
                the task. If 'instance' is provided, the created task
                instance will be returned.
            args_preserve (list, optional): Set of argument names to
                preserve following a run.
            cache_outputs (list, optional): Set of outputs that should be
                cached.
            **kwargs: Additional keyword arguments are passed to
                from_external_args.

        Returns:
            object: Result of the run.

        """
        assert 'root' not in kwargs
        self = cls.from_external_args(args, **kwargs)
        out = self.run(output_name=output_name)
        if args_preserve:
            for k in args_preserve:
                setattr(args, k, getattr(self.args, k))
        if cache_outputs:
            assert 'root' in kwargs or 'cached_outputs' in kwargs
            self.cache_output(cache_outputs)
        return out

    def copy_matching_outputs(self, other):
        r"""Copy existing outputs from another task to prevent repeated
        I/O when the files match.

        Args:
            other (TaskBase): Instance to copy from.

        """
        assert isinstance(other, type(self))
        copied = []
        for k in self._outputs_local:
            self_fname = self.output_file(k, return_disabled=True)
            other_fname = other.output_file(k, return_disabled=True)
            if (((not self_fname)
                 or self_fname != other_fname
                 or k in self._outputs
                 or k not in other._outputs)):
                continue
            self._outputs[k] = other._outputs[k]
            copied.append(k)
        for k, v in self.external_tasks.items():
            vother = other.external_tasks[k]
            v.copy_matching_outputs(vother)

    def run_series(self, cls=None, **kwargs):
        r"""Run the process for a series of arguments.

        Args:
            cls (type, optional): Task class that should be run in
                iteration. Defaults to the type of the current task.
            **kwargs: Additional keyword arguments are passed to
                cls.run_series_class.

        Yields:
            object: Results from each step.

        """
        if cls is None:
            cls = type(self)
        kwargs.setdefault('cached_outputs', self._cached_outputs)
        for x in cls.run_series_class(self.args, **kwargs):
            yield x

    @classmethod
    def run_series_class(cls, args, over=None, per_iter=None,
                         args_overwrite=None, **kwargs):
        r"""Run the process for a series of arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
            over (dict, optional): Mapping between argument names and
                values that should be iterated over.
            per_iter (dict, optional): Dictionary of argument values that
                should be added for each iteration. This can be updated
                between iterations.
            args_overwrite (dict, optional): Arguments to overwrite.
            **kwargs: Additional keyword arguments are passed to
                cls.run_iteration.

        Yields:
            object: Results from each step.

        """
        if per_iter is None:
            per_iter = {}
        if args_overwrite is None:
            args_overwrite = {}
        if not over:
            yield cls.run_iteration_class(
                args, args_overwrite=dict(args_overwrite, **per_iter),
                **kwargs
            )
            return
        keys = list(over.keys())
        for k in keys:
            if not isinstance(over[k], list):
                over[k] = [over[k]]
        cls.log_class(f"STARTING LOOP OVER: {keys}")
        for x in itertools.product(*[over[k] for k in keys]):
            iargs_overwrite = dict(args_overwrite, **per_iter)
            iargs_overwrite[f'output_{cls._name}'] = True
            for k, v in zip(keys, x):
                iargs_overwrite[k] = v
            cls.log_class(f'ITERATION: {dict(zip(keys, x))}')
            yield cls.run_iteration_class(
                args, args_overwrite=iargs_overwrite,
                **kwargs
            )

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


class IterationTaskBase(TaskBase):
    r"""Base class for iterating over a task."""

    _step_task = None
    _step_vary = None

    def __init__(self, *args, **kwargs):
        self._step_results = []
        super(IterationTaskBase, self).__init__(*args, **kwargs)

    @staticmethod
    def _on_registration(cls):
        if cls._step_task is not None:
            step_prop = cls._external_tasks.get(cls._step_task, {})
            if cls._step_vary is not None:
                step_prop.setdefault('exclude', [])
                if cls._step_vary not in step_prop['exclude']:
                    step_prop['exclude'].append(cls._step_vary)
            cls._external_tasks = copy.deepcopy(cls._external_tasks)
            cls._external_tasks[cls._step_task] = step_prop
            cls._output_info.setdefault(cls._name, {})
            cls._output_info[cls._name].setdefault(
                'base', cls._step_task._name)
        TaskBase._on_registration(cls)

    def overwrite_outputs(self, downstream=None, wildcards=None):
        r"""Remove existing files that should be overwritten.

        Args:
            downstream (list, optional): Existing list that output names
                should be added to that downstream files need to be
                removed for.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file names.

        """
        if wildcards is None:
            wildcards = []
        if ((self._step_vary is not None
             and self._step_vary not in wildcards)):
            wildcards = wildcards + [self._step_vary]
        # for k in self._step_task._outputs_total:
        #     if not getattr(self.args, f'overwrite_{k}', False):
        #         continue
        #     step_task = self._get_output_task(self, k)
        #     args_base = self.argument_names()
        #     args_step = step_task.argument_names()
        #     wildcards = list(set(args_step) - set(args_base))
        #     print("OVERWRITE", k, wildcards)
        #     pdb.set_trace()
        #     if self._step_vary is not None:
        #         assert self._step_vary in wildcards
        #     if vary is not None:
        #         wildcards += [x for x in vary if x not in wildcards]
        #     self.remove_output(k, wildcards=wildcards)
        super(IterationTaskBase, self).overwrite_outputs(
            downstream=downstream, wildcards=wildcards,
        )

    def step_args(self):
        r"""Yield the updates that should be made to the arguments for
        each step.

        Yields:
            dict: Step arguments.

        """
        raise NotImplementedError

    def finalize_step(self, x):
        r"""Finalize the output from a step.

        Args:
            x (object): Result of step.

        Returns:
            object: Finalized step result.

        """
        return x

    def join_steps(self, xlist):
        r"""Join the output form all of the steps.

        Args:
            xlist (list): Result of all steps.

        Returns:
            object: Joined output from all steps.

        """
        return xlist

    def run_steps(self, output_name='instance'):
        r"""Run the steps.

        Args:
            output_name (str, optional): Step output that should be
                passed to finalize_step for each step.

        Returns:
            list: Output from each step.

        """
        self._step_results = []
        x_prev = None
        for args_overwrite in self.step_args():
            iargs_overwrite = copy.deepcopy(args_overwrite)
            pprint.pprint(iargs_overwrite)
            while True:
                try:
                    x = self.run_iteration(
                        cls=self._step_task,
                        args_overwrite=iargs_overwrite,
                        output_name=output_name,
                        copy_outputs_from=x_prev,
                    )
                    self._step_results.append(self.finalize_step(x))
                    if isinstance(x, TaskBase):
                        del x_prev
                        x_prev = x
                        gc.collect()
                    break
                except RepeatIterationError as e:
                    iargs_overwrite.update(e.args_overwrite)
        return self._step_results

    def _generate_output(self, name):
        r"""Generate the specified output value.

        Args:
            name (str): Name of the output to generate.

        Returns:
            object: Generated output.

        """
        if name != self._name:
            return super(IterationTaskBase, self)._generate_output(name)
        if self._step_task is None:
            raise NotImplementedError
        if hasattr(self._step_task, 'adjust_args_step'):
            self._step_task.adjust_args_step(self.args, self._step_vary)
        return self.join_steps(self.run_steps())


class OptimizationTaskBase(IterationTaskBase):
    r"""Base class for tasks that iterate to achieve a result."""

    _final_outputs = []
    _arguments = [
        (('--vary', ), {
            'type': str,
            'help': 'Argument that should be varied.'
        }),
        (('--goal', ), {
            'help': 'Goal of the optimization.',
        }),
        (('--method', ), {
            'type': str, 'choices': ['nelder-mead', 'powell'],
            'default': 'nelder-mead',
            'help': (
                'Method that should be used to minimize the objective'
            ),
        }),
        (('--tolerance', ), {
            'type': float, 'default': 1e-5,
            'help': 'Tolerance for achieving result',
        }),
    ]
    _output_suffix = SuffixGenerator([
        ('goal', {}),
        ('vary', {'prefix': '_vs_'}),
        ('method', {'noteq': 'nelder-mead'}),
    ])

    @staticmethod
    def _on_registration(cls):
        for k in cls._final_outputs:
            kcls = cls.get_output_task(k)
            cls._output_info.setdefault(
                k, {
                    'base': cls._name,
                    'ext': kcls._output_info[k]['ext'],
                    'description': kcls._output_info[k]['description'],
                    'optional': True,
                }
            )
        IterationTaskBase._on_registration(cls)
        if cls._step_task is not None:
            cls._output_info[cls._name].setdefault('ext', '.json')

    def overwrite_outputs(self, downstream=None, wildcards=None):
        r"""Remove existing files that should be overwritten.

        Args:
            downstream (list, optional): Existing list that output names
                should be added to that downstream files need to be
                removed for.
            wildcards (list, optional): List of arguments that wildcards
                should be used for in the generated output file names.

        """
        if wildcards is None:
            wildcards = []
        if self.args.vary not in wildcards:
            wildcards = wildcards + [self.args.vary]
        super(OptimizationTaskBase, self).overwrite_outputs(
            downstream=downstream, wildcards=wildcards,
        )

    @cached_property
    def goal(self):
        r"""units.Quantity: Value that should be achieved."""
        if self.args.goal in ['minimize', 'maximize']:
            return self.args.goal
        # TODO: Fix units
        return parse_quantity(self.args.goal)

    @cached_property
    def goal_units(self):
        r"""str: Goal units."""
        if isinstance(self.goal, units.Quantity):
            return str(self.goal.units)
        return None

    @cached_property
    def vary_units(self):
        r"""str: Units of argument to vary."""
        x = getattr(self.args, self.args.vary)
        if isinstance(x, units.Quantity):
            return str(x.units)
        return None

    def objective(self, x):
        r"""Objective function for use with scipy.optimize.minimize.

        Args:
            x (np.ndarray): Input arguments.

        Returns:
            float: Result.

        """
        assert len(x) == 1
        iargs_overwrite = {
            self.args.vary: parse_quantity(x[0], self.vary_units)
        }
        # TODO: Does this work?
        for k in self._step_task._outputs_total:
            iargs_overwrite[f'dont_write_{k}'] = True
        repeat = True
        out = None
        while repeat:
            try:
                self.log(f'OBJECTIVE:\n{pprint.pformat(iargs_overwrite)}',
                         force=True)
                x = self.run_iteration(
                    cls=self._step_task,
                    args_overwrite=iargs_overwrite,
                    copy_outputs_from=self._prev_instance,
                )
                result = self.finalize_step(x)
                if self.goal == 'minimize':
                    out = result
                elif self.goal == 'maximize':
                    out = -result
                else:
                    out = np.abs(float(
                        (result - self.goal) / self.goal))
                self.log(f'OBJECTIVE RESULT: {out} ({result})',
                         force=True)
                self._prev_instance = x
                repeat = False
            except RepeatIterationError as e:
                iargs_overwrite.update(e.args_overwrite)
        return out

    def run_steps(self, output_name='instance'):
        r"""Run the steps.

        Args:
            output_name (str, optional): Step output that should be
                passed to finalize_step for each step.

        Returns:
            list: Output from each step.

        """
        from scipy.optimize import minimize
        self.goal  # Initialize
        x0 = np.array([getattr(self.args, self.args.vary)])
        self._prev_instance = None
        res = minimize(
            self.objective, x0, method=self.args.method,
            tol=self.args.tolerance,
            # options={
            #     'xatol': self.args.tolerance,
            # }
        )
        assert res.success
        out = {
            self.args.vary: res.x[0],
        }
        return out

    def final_output_args(self, name):
        r"""Get the arguments that should be used generate the final
        output.

        Args:
            name (str): Name of the final output to generate.

        Returns:
            dict: Arguments to use.

        """
        return {}

    def _generate_output(self, name, output_name=None):
        r"""Generate the specified output value.

        Args:
            name (str): Name of the output to generate.

        Returns:
            object: Generated output.

        """
        if name in self._final_outputs:
            if output_name is None:
                output_name = name
            cls = self.get_output_task(name)
            args_overwrite = dict(
                self.get_output(self._name),
                **self.final_output_args(name)
            )
            return self.run_iteration(
                cls=cls,
                args_overwrite=args_overwrite,
                output_name=output_name,
            )
        return super(OptimizationTaskBase, self)._generate_output(name)


class TemporalTaskBase(IterationTaskBase):
    r"""Base class for tasks that iterate over another class."""

    _step_task = None
    _step_vary = 'time'
    _composite_arguments = {
        'time': {
            'start_time': {
                'description': ' to start at.',
                'defaults': {'hour': 'sunrise'},
            },
            'stop_time': {
                'description': ' to stop at.',
                'defaults': {'hour': 'sunset'},
                'name_base': 'start_time',
            },
        }
    }
    _arguments = [
        (('--duration', ), {
            'type': parse_quantity, 'units': 'hours',
            'help': 'The time that the animation should last',
        }),
        (('--step-count', ), {
            'type': int,
            'help': ('The number of time steps that should be taken '
                     'between the start and end time. If not provided, '
                     'the number of time steps will be determined from '
                     '\"step_interval\"'),
        }),
        (('--step-interval', ), {
            'type': parse_quantity, 'units': 'hours',
            'help': ('The interval (in hours) that should be used '
                     'between time steps. If not provided, '
                     '\"step_count\" will be used to calculate the '
                     'step interval. If \"step_count\" is not '
                     'provided, a step interval of 1 hour will be '
                     'used.'),
        }),
        (('--dont-age', ), {
            'action': 'store_true',
            'help': (
                'Don\' age the generated scene. Currently this is set '
                'to true if the start and end date are the same, but '
                'this may change in the future.'
            )
        }),
    ]
    _output_suffix = SuffixGenerator([
        ('stop_time', {}),
    ])

    @classmethod
    def adjust_args_internal(cls, args):
        r"""Adjust the parsed arguments including setting defaults that
        depend on other provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.

        """
        cls._convert_composite(args, 'start_time', base='time')
        if not TimeArgument.any_arguments_set(args, 'stop_time'):
            if args.duration is None and (args.step_interval is None
                                          or args.step_count is None):
                if args.start_time.crop_age_string == 'planting':
                    assert not args.dont_age
                    args.stop_age = 'maturity'
                    args.planting_date = args.start_time.date
                    if args.start_time.solar_time_string:
                        args.stop_hour = (
                            args.start_time.solar_time_string
                        )
                elif args.start_time.solar_time_string == 'sunrise':
                    args.stop_hour = 'sunset'
                    args.stop_date = args.start_time.date
                    if args.start_time.crop_age_string:
                        args.stop_age = (
                            args.start_time.crop_age_string
                        )
                else:
                    args.duration = units.Quantity(24.0, 'hours')
            else:
                if args.duration is None:
                    args.duration = (
                        args.step_interval * args.step_count
                    )
                args.stop_time = (
                    args.start_time.time
                    + utils.quantity2timedelta(args.duration)
                )
        super(TemporalTaskBase, cls).adjust_args_internal(args)
        if args.stop_time.time == args.start_time.time:
            args.stop_time = args.stop_time.time.replace(
                hour=0, minute=0, microsecond=0)
            cls._convert_composite(args, 'stop_time', base='time',
                                   overwrite=True)
        assert args.stop_time.time > args.start_time.time
        duration = args.stop_time.time - args.start_time.time
        duration = utils.timedelta2quantity(duration)
        if not args.step_count:
            if not args.step_interval:
                ndays = int(duration.to('days'))
                if ndays > 2:
                    args.step_interval = units.Quantity(1.0, 'days')
                else:
                    args.step_interval = units.Quantity(1.0, 'hours')
            args.step_count = int(duration / args.step_interval)
        elif not args.step_interval:
            args.step_interval = duration / args.step_count
        args.start_time.update_args(args, name='time')
        if args.start_time.date == args.stop_time.date:
            args.stop_time.ignore_date = True
            # TODO: Update this?
            args.dont_age = True

    def step_args(self):
        r"""Yield the updates that should be made to the arguments for
        each step.

        Yields:
            dict: Step arguments.

        """
        dt = self.args.step_interval
        iargs = None
        for i in range(self.args.step_count):
            iargs = self.args.start_time.iteration_args(
                i * dt, dont_age=self.args.dont_age)
            yield iargs
        time = TimeArgument.from_kwargs(None, iargs)
        if time.time < self.args.stop_time.time:
            yield self.args.stop_time.iteration_args()


#################################################################
# CLI
#################################################################

def parse(**kwargs):
    r"""Parse arguments provided via the command line or keyword
    arguments

    Args:
        **kwargs: If any keyword args are passed, they are parsed
            instead of the command line arguments.

    Returns:
        argparse.Namespace, dict: Argument namespace and keyword
            keyword arguments that were not parsed.

    """
    parser = InstrumentedParser("Generate/analyze a 3D canopy model")
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
    if kwargs:
        raise AssertionError(f'Unparsed kwargs: {pprint.pformat(kwargs)}')
    args.SUBPARSER_CLASS = parser.subparser_class('task', args)
    return args


def main(**kwargs):
    r"""Parse arguments provided via the command line or keyword
    arguments and run the parsed task.

    Args:
        **kwargs: If any keyword args are passed, they are parsed
            instead of the command line arguments.

    Returns:
        argparse.Namespace, dict: Argument namespace and keyword
            keyword arguments that were not parsed.

    """
    args = parse(**kwargs)
    inst = args.SUBPARSER_CLASS(args)
    return inst.run(output_name='instance')


if __name__ == "__main__":
    main()
