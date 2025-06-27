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
    parse_quantity, format_list_for_help,
    get_class_registry, NoDefault, RegisteredClassBase,
    cached_property, cached_factory_property,
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

    # def __init__(self, parent, members):
    #     self.parent = parent
    #     self.members = members

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
            '--show-irrelevant', default=0,
            dest='show_irrelevant', type=int, const=-1, nargs='?',
            help=('Show arguments that are diabled by absent or '
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
                    if args.show_irrelevant > 0:
                        if i >= args.show_irrelevant:
                            out = True
                        add_to_parent = (i <= args.show_irrelevant)
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
        if args.show_irrelevant == -1:
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

    def run_subparser(self, group, args):
        r"""Run the subparser selected by a subparser group.

        Args:
            group (str): Name of the subparser group to run.
            args (argparse.Namespace): Parsed arguments.

        """
        subparser = self.get_subparsers(group)
        subparser_name = getattr(args, group)
        if subparser_name is None:
            subparser_name = subparser.default
        func = subparser.func
        if func is None:
            func = get_class_registry().get(group, subparser_name)
        print(f"RUNNING {subparser_name}")
        func(args)

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
        # if subparsers:
        #     import pprint
        #     pprint.pprint(args)
        #     pprint.pprint(kwargs)
        #     print('subparsers', subparsers)
        #     import pdb; pdb.set_trace()
        assert not subparsers
        assert not subparser_options
        # for x in self.subparsers(subparsers=subparsers):
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
        self.run(top_level=True, **kwargs)

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

    @staticmethod
    def add_arguments_static(cls, parser, only_subparser=False,
                             exclude=None, include=None):
        r"""Add arguments associated with this subparser to a parser.

        Args:
            parser (InstrumentedParser): Parser that the arguments
                should be added to.
            only_subparser (bool, optional): If True, only add the
                subparser if it is missing.
            exclude (list, optional): Set of arguments to exclude.
            include (list, optional): Set of arguments to include.

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
        if only_subparser:
            return
        for iargs, ikwargs in cls._arguments:
            if exclude is not None or include is not None:
                idest = ikwargs.get(
                    'dest', iargs[0].lstrip('-').replace('-', '_'))
                if exclude is not None and idest in exclude:
                    continue
                if include is not None and idest not in include:
                    continue
            # ikwargs.setdefault('subparsers', {})
            # ikwargs['subparsers'].setdefault(cls._registry_key,
            #                                  [cls._name])
            # parser.add_argument(*iargs, **ikwargs)
            subparser.add_argument(*iargs, **ikwargs)

    @classmethod
    def add_arguments(cls, parser, **kwargs):
        r"""Add arguments associated with this subparser to a parser.

        Args:
            parser (InstrumentedParser): Parser that the arguments
                should be added to.
            **kwargs: Additional keyword arguments are passed to the
                add_arguments_static method.

        """
        return cls.add_arguments_static(cls, parser, **kwargs)

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
        ival = None
        if ikwargs.get('action', None) == 'store_true':
            ival = False
        elif ikwargs.get('action', None) == 'store_false':
            ival = True
        elif 'default' in ikwargs:
            ival = ikwargs['default']
            if 'type' in ikwargs and ival is not None:
                ival = ikwargs['type'](ival)
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
        raise Exception
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
                  require_alternate_output=None,
                  top_level=False, **kwargs):
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
            top_level (bool, optional): If True, this is the top level
                call and arguments should not be cached.
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
        if top_level:
            assert not any([
                args_preserve, args_overwrite, properties_preserve,
                properties_overwrite, require_alternate_output,
                recursive,
            ])
        else:
            self.cache_args(
                adjust=cls, args_preserve=args_preserve,
                args_overwrite=args_overwrite,
                properties_preserve=properties_preserve,
                properties_overwrite=properties_overwrite,
                alternate_outputs=require_alternate_output,
                recursive=recursive,
            )
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
        if not top_level:
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
    parser.run_subparser('task', args)
    return args, kwargs


if __name__ == "__main__":
    args, _ = main()
