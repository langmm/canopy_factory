import pprint
import pdb
import copy
import functools
import numpy as np
import pandas as pd
import scipy
import contextlib
from abc import abstractmethod
from collections.abc import MutableMapping
import openalea.plantgl.all as pgl
import openalea.plantgl.math as pglmath
from openalea.plantgl.math import Vector2, Vector3, Vector4
from openalea.plantgl.scenegraph import (
    NurbsCurve2D, NurbsCurve, NurbsPatch)
from yggdrasil import rapidjson
from canopy_factory.utils import (
    RegisteredMetaClass, RegisteredClassBase, get_class_registry,
    NoDefault, jsonschema2argument, format_list_for_help, UnitSet,
)


class ImmutableDictException(BaseException):
    pass


class DictWrapper(MutableMapping):
    r"""Abstract base class for dictionary wrappers."""

    def __init__(self, logger=None, logger_prefix=''):
        self.logger = logger
        self._logger_prefix = logger_prefix
        self._original_keys = set(self.keys(raw=True))
        # TODO: Remove this check
        # allkeys = list(self.keys())
        # if len(allkeys) != len(set(allkeys)):
        #     pdb.set_trace()
        # assert len(allkeys) == len(set(allkeys))

    @property
    @abstractmethod
    def dest(self):
        r"""DictWrapper: Destination dictionary for added keys."""
        raise ImmutableDictException("Immutable")

    @property
    def mutable(self):
        r"""bool: True if keys can be added to the dictionary."""
        try:
            self.dest
            return True
        except ImmutableDictException:
            return False

    @property
    def flattened(self):
        r"""dict: Flattened version of the members."""
        return dict(self.items())

    @property
    def raw_flattened(self):
        r"""dict: Copy of raw destination dictionary (with prefix)"""
        return dict(self.items(raw=True))

    def __repr__(self):
        return self.flattened.__repr__()

    def __len__(self):
        return len(list(self._get_iterator()))

    def __iter__(self):
        for k, v in self._get_iterator():
            yield k

    def keys(self, raw=False):
        r"""Wrapped dictionary keys with prefixes removed.

        Args:
            raw (bool, optional): If True, the raw dictionary keys will
                be returned without the prefix removed.

        Returns:
            dict_keys: Keys view.

        """
        if not raw:
            return super(DictWrapper, self).keys()
        return [k for k, v in self._get_iterator(raw=raw)]

    def items(self, raw=False):
        r"""Wrapped dictionary items with prefixes removed from keys.

        Args:
            raw (bool, optional): If True, the raw dictionary items will
                be returned without the prefix removed.

        Returns:
            dict_items: Items view.

        """
        if not raw:
            return super(DictWrapper, self).items()
        return [x for x in self._get_iterator(raw=raw)]

    def setdefault(self, k, v):
        if k in self:
            return
        self[k] = v

    @property
    def added(self):
        r"""set: Keys added to the dictionary."""
        return set(self.keys()) - self._original_keys

    @property
    def removed(self):
        r"""set: Keys removed from the dictionary."""
        return self._original_keys - set(self.keys())

    @classmethod
    def select_keys(cls, x, prefix=''):
        return {
            k: copy.deepcopy(v) for k, v in x.items()
            if (not prefix) or k.startswith(prefix)
        }

    @classmethod
    def remove_prefixed_keys(cls, x, prefix):
        for k in list(x.keys()):
            if k.startswith(prefix):
                del x[k]

    def _forward_key(self, k):
        return k

    def _reverse_key(self, k):
        return k

    def copy_src2dst(self, kdst, ksrc=NoDefault, overwrite=False):
        r"""Copy a key/value pair from the source dictionary to the
        destination dictionary if it is present in the source dictionary
        and not present in the destination dictionary.

        Args:
            kdst (str): Key that should be assigned to in the destination.
            ksrc (str, optional): Key that should be copyied from the
                source dictionary. If not provided, kdst is used.
            overwrite (bool, optional): If True, overwrite any existing
                value in the destination dictionary.

        """
        if ksrc == NoDefault:
            ksrc = kdst
        val = NoDefault
        if ksrc in self:
            val = self[ksrc]
        elif kdst in self:
            val = self[kdst]
        kdst_raw = self._forward_key(kdst)
        if (overwrite or kdst_raw not in self.dest) and val is not NoDefault:
            self.dest[kdst_raw] = val

    def remove_cond(self, fcond):
        r"""Remove key/value pairs from this dictionary based on the
        value of a provided function.

        Args:
            fcond (callable): Function that takes a single raw key/value
                pair and returns True when the pair should be removed.

        """
        for k, v in list(self.items(raw=True)):
            if not fcond(k, v):
                del self[self._reverse_key(k)]

    def select_cond(self, fcond, transform=False, deepcopy=False):
        r"""Create a new PrefixedDict instance that only includes
        key/value pairs selected by the provided function.

        Args:
            fcond (callable): Selection function that takes a single
                key/value pair and returns True or False to indicate
                if the pair is selected.
            transform (bool, callable, optional): A function that should
                be used to transform raw keys for inclusion in the
                returned dictionary. If True, self._reverse_key will be
                used.
            deepcopy (bool, optional): If True, deepcopy values.

        Returns:
            dict: Dictionary containing selected members.

        """
        if transform is True:
            transform = self._reverse_key
        wrapped = {}
        for k, v in self.items(raw=True):
            if fcond(k, v):
                kx = transform(k) if transform else k
                if deepcopy:
                    wrapped[kx] = copy.deepcopy(v)
                else:
                    wrapped[kx] = v
        return wrapped

    def remove_prefix(self, prefix, **kwargs):
        r"""Remove keys that start with the provided prefix.

        Args:
            prefix (str): Prefix to remove keys with.
            **kwargs: Additional keyword arguments are passed to
                remove_cond.

        """

        def fcond(k, v):
            return ((not prefix) or k.startswith(prefix))

        self.remove_cond(fcond, **kwargs)

    def select_prefix(self, prefix, strip=False, **kwargs):
        r"""Create a new PrefixedDict instance that only includes keys
        that start with a given prefix.

        Args:
            prefix (str): Prefix to filter on.
            strip (bool, optional): If True, the prefix should be
                stripped in the returned dictionary.
            **kwargs: Additional keyword arguments are passed to
                select_cond.

        Returns:
            dict: Dictionary containing selected members.

        """

        def fcond(k, v):
            return ((not prefix) or k.startswith(prefix))

        def fstrip(k):
            if not prefix:
                return k
            return k.split(prefix, 1)[-1]

        if strip:
            assert 'transform' not in kwargs
            kwargs['transform'] = fstrip
        return self.select_cond(fcond, **kwargs)

    def strip_prefix(self, prefix, keys=None):
        r"""Remove a prefix from keys in the dictionary.

        Args:
            prefix (str): Prefix to strip.
            keys (list, optional): Set of keys to remove the prefix from.
                If not provided, all keys will be used and an error will
                be raised if any keys do not start with the prefix.

        """
        if not prefix:
            return
        if keys is None:
            keys = list(self.keys())
        for k in keys:
            if not k.startswith(prefix):
                raise KeyError(f"Key \"{k}\" does not "
                               f"start with prefix \"{prefix}\"")
            ky = k.split(prefix, 1)[-1]
            self[ky] = self.pop(k)

    def update_missing(self, other):
        r"""Update the dictionary only will values that are not present.

        Args:
            other (dict): Values to update the dictionary with.

        """
        for k, v in other.items():
            self.setdefault(k, v)

    @classmethod
    def coerce(cls, x, **kwargs):
        r"""Coerce an object to be a DictWrapper compatible instance.

        Args:
            x (dict, DictWrapper, tuple, list): Object that can be
                interpreted as a DictWrapper instance.
            **kwargs: Additional keyword arguments are passed to the
                constructor for the interpreted instance.

        Returns:
            DictWrapper: Coerced instance.

        """
        if isinstance(x, tuple):
            assert len(x) == 2
            return PrefixedDict(x, **kwargs)
        elif isinstance(x, list):
            x = DictSet(x)
        if isinstance(x, (dict, DictWrapper)):
            if kwargs:
                return PrefixedDict(x, **kwargs)
            return x
        raise TypeError(type(x))

    @abstractmethod
    def _get_iterator(self, raw=False, prev=None):
        r"""Yields member key/item pairs.

        Args:
            raw (bool, optional): If True, raw keys should be yielded.
            prev (list, optional): Set of keys already yielded.

        Yields:
            tuple: Member items.

        """
        raise NotImplementedError

    @contextlib.contextmanager
    def temporary_prefix(self, prefix, append=False, report_change=False):
        r"""Temporarily set a prefix for the dictionary with the context.
        When the context exits, any previous prefix will be restored and
        any added parameters will be maintained.

        Args:
            prefix (str): Prefix to use within the context.
            append (bool, optional): If True, the prefix should be
                appended to the end of the current prefix.
            report_change (str, bool, optional): If True, changes
                that occur in the context will be reported. If a string
                is provided, it will be used as a prefix for the log
                message.

        Yields:
            DictWrapper: View of the dictionary with the provided prefix.

        """
        assert not append
        if report_change is True:
            report_change = ''
        if isinstance(report_change, str):
            cm = self.report_change(prefix=report_change)
        else:
            cm = contextlib.nullcontext()
        with cm:
            yield PrefixedDict(self, prefix=prefix)

    @classmethod
    def assert_keys_match(cls, x, keys0, logger=None):
        if not list(x.keys()) == keys0:
            message = (
                f'ADDED:   {set(x.keys()) - set(keys0)}\n'
                f'REMOVED: {set(keys0) - set(x.keys())}'
            )
            if logger:
                logger.log(message, force=True)
            else:
                print(message)
            pdb.set_trace()
        assert list(x.keys()) == keys0

    @property
    def logger_prefix(self):
        r"""str: Prefix for log messages."""
        return self._logger_prefix

    def report_diff(self, keys0, keys1, prefix='', suffix=''):
        r"""Report on a difference between two key sets.

        Args:
            keys0 (set): Set of keys from a prior state.
            keys1 (set): Set of keys from a state after keys0.
            prefix (str, optional): Prefix for report message.
            suffix (str, optional): Suffix for report message.

        """
        added = keys1 - keys0
        removed = keys0 - keys1
        msg = ''
        if added:
            msg += f'ADDED {added}.'
        if removed:
            msg += ' ' if added else ''
            msg += f'REMOVED {removed}.'
        if not (added or removed):
            msg += 'NO CHANGE'
            return
        self.logger.log(f'{self.logger_prefix}[{self.prefix}]: '
                        f'{prefix}{msg}{suffix}')

    @contextlib.contextmanager
    def report_change(self, prefix='', suffix='',
                      assert_no_change=False, **kwargs):
        r"""Report on how the dictionary changed within the context.

        Args:
            prefix (str, optional): Prefix for report message.
            suffix (str, optional): Suffix for report message.
            assert_no_change (bool, optional): If True, assert that the
                keys did not change.
            **kwargs: Additional keyword arguments are passed to the
                keys method to get the set of keys before & after the
                context.

        """
        kwargs.setdefault('raw', True)
        keys0 = set(self.keys(**kwargs))
        try:
            yield
        finally:
            keys1 = set(self.keys(**kwargs))
            if assert_no_change:
                assert keys0 == keys1
            self.report_diff(keys0, keys1, prefix=prefix, suffix=suffix)


pprint.PrettyPrinter._dispatch[DictWrapper.__repr__] = (
    pprint.PrettyPrinter._pprint_dict
)


class PrefixedDict(DictWrapper):
    r"""Dictionary wrapper that allows for adding a prefix to keys when
    accessing the wrapped dictionary.

    Args:
        wrapped (dict): Wrapped dictionary.
        prefix (str, optional): Prefix to add to keys.
        immutable (bool, optional): If True, the dictionary should not
            be modified.
        **kwargs: Additional keyword arguments are passed to the parent
            constructor.

    """

    def __init__(self, wrapped, prefix=None, immutable=False, **kwargs):
        if wrapped is None:
            wrapped = {}
        if isinstance(wrapped, tuple):
            if prefix is None:
                prefix = wrapped[0]
                wrapped = wrapped[1]
            else:
                wrapped = PrefixedDict(wrapped[1], prefix=wrapped[0])
        if prefix is None:
            prefix = ''
        self._wrapped = wrapped
        self._immutable = immutable
        self.prefix = prefix
        super(PrefixedDict, self).__init__(**kwargs)

    @property
    def logger_prefix(self):
        r"""str: Prefix for log messages."""
        return f'{self._logger_prefix}[{self.prefix}]'

    @property
    def dest(self):
        r"""DictWrapper: Destination dictionary for added keys."""
        if self._immutable:
            raise ImmutableDictException("Immutable")
        return self._wrapped

    def _get_iterator(self, raw=False, prev=None):
        r"""Yields member key/item pairs.

        Args:
            raw (bool, optional): If True, raw keys should be yielded.
            prev (list, optional): Set of keys already yielded.

        Yields:
            tuple: Member items.

        """
        if prev is None:
            prev = []
        for k, v in self._wrapped.items():
            try:
                kx = self._reverse_key(k)
            except KeyError:
                continue
            if kx in prev:
                continue
            if raw:
                yield (k, v)
            else:
                yield (kx, v)
            prev.append(kx)

    def _forward_key(self, k):
        if not self.prefix:
            return k
        return f'{self.prefix}{k}'

    def _reverse_key(self, k):
        if not self.prefix:
            return k
        if not k.startswith(self.prefix):
            raise KeyError(f'Key \"{k}\" does not start with prefix '
                           f'\"{self.prefix}\"')
        try:
            assert k.startswith(self.prefix)
        except AssertionError:
            print(k, self.prefix)
            pdb.set_trace()
            raise
        return k.split(self.prefix, 1)[-1]

    def __getitem__(self, k):
        return self._wrapped[self._forward_key(k)]

    def __setitem__(self, k, v):
        self.dest[self._forward_key(k)] = v

    def __delitem__(self, k):
        del self.dest[self._forward_key(k)]
        assert k not in self

    def select_prefix(self, prefix=None, strip=False, **kwargs):
        r"""Create a new PrefixedDict instance that only includes keys
        that start with a given prefix.

        Args:
            prefix (str, optional): Prefix to filter on. If not provided,
                the prefix for the target will be used.
            strip (bool, optional): If True, the prefix should be
                stripped in the returned dictionary.
            **kwargs: Additional keyword arguments are passed to the
                parent method.

        Returns:
            PrefixedDict: Dictionary containing selected members.

        """
        if prefix is None:
            prefix = self.prefix
        out = super(PrefixedDict, self).select_prefix(prefix, **kwargs)
        out_prefix = prefix if not kwargs.get('strip', False) else ''
        return PrefixedDict(out, prefix=out_prefix)

    @contextlib.contextmanager
    def temporary_prefix(self, prefix, append=False, report_change=False):
        r"""Temporarily set a prefix for the dictionary with the context.
        When the context exits, any previous prefix will be restored and
        any added parameters will be maintained.

        Args:
            prefix (str): Prefix to use within the context.
            append (bool, optional): If True, the prefix should be
                appended to the end of the current prefix.
            report_change (str, bool, optional): If True, changes
                that occur in the context will be reported. If a string
                is provided, it will be used as a prefix for the log
                message.

        Yields:
            DictWrapper: View of the dictionary with the provided prefix.

        """
        # if isinstance(self._wrapped, DictWrapper) and not self.prefix:
        #     old_wrapped = self._wrapped
        #     with (self._wrapped.temporary_prefix(prefix, append=append)
        #           as new_wrapped):
        #         self._wrapped = new_wrapped
        #         try:
        #             yield self
        #         finally:
        #             self._wrapped = old_wrapped
        #     return
        if append:
            prefix = self.prefix + prefix
        if report_change is True:
            report_change = ''
        if isinstance(report_change, str):
            cm = self.report_change(prefix=report_change)
        else:
            cm = contextlib.nullcontext()
        old_prefix = self.prefix
        with cm:
            self.prefix = prefix
            try:
                yield self
            finally:
                self.prefix = old_prefix

    # Methods for nested DictSet
    @property
    def members(self):
        r"""list: Members of the wrapped dictionary. Only valid if
        the wrapped dictionary is a DictSet."""
        return self._wrapped.members

    @members.setter
    def members(self, x):
        self._wrapped.members = x

    def insert(self, idx, member, add_prefix=False, **kwargs):
        r"""Add a new member to the set.

        Args:
            idx (index): Index to insert the new member add.
            member (dict): New member.
            add_prefix (str, bool, optional): Prefix that should be added
                to the keys in x. If True, the current prefix should be
                added.
            **kwargs: Additional keyword arguments are passed to
                coerce_member.

        """
        if add_prefix is True:
            add_prefix = self.prefix
        return self._wrapped.insert(idx, member, add_prefix=add_prefix,
                                    **kwargs)

    def append(self, member, add_prefix=False, **kwargs):
        r"""Add a new member to the end of the set.

        Args:
            member (dict): New member.
            add_prefix (str, bool, optional): Prefix that should be added
                to the keys in x. If True, the current prefix should be
                added.
            **kwargs: Additional keyword arguments are passed to
                coerce_member.

        """
        if add_prefix is True:
            add_prefix = self.prefix
        return self._wrapped.append(member, add_prefix=add_prefix,
                                    **kwargs)


@contextlib.contextmanager
def temporary_prefix(x, prefix, **kwargs):
    r"""Create a wrapper around a dictionary with a temporary prefix that
    should be added when accessing dictionary elements.

    Args:
        x (dict, DictWrapper): Dictionary to wrap with a prefix.
        prefix (str): Prefix to use within the context.
        **kwargs: Additional keyword arguments are passed to the
            temporary_prefix PrefixedDict method.

    Yields:
        PrefixedDict: Dictionary with prefix wrapper.

    """
    if not isinstance(x, DictWrapper):
        x = PrefixedDict(x)
    with x.temporary_prefix(prefix, **kwargs):
        yield x


class DictSet(DictWrapper):
    r"""Container for chaining multiple dictionaries.

    Args:
        members (list): Set of dictionaries that should be accessed in
            the order provided.
        **kwargs: Additional keyword arguments are passed to the parent
            constructor.

    """

    def __init__(self, members, **kwargs):
        assert isinstance(members, list)
        self.members = []
        for x in members:
            self.append(x)
        super(DictSet, self).__init__(**kwargs)

    @classmethod
    def coerce_member(cls, x, add_prefix=False, **kwargs):
        r"""Coerce an object to a DictWrapper instance.

        Args:
            x (object): Dictionary-like object.
            add_prefix (str, optional): Prefix that should be added to the
                keys in x.
            **kwargs: Additional keyword arguments are passed to
                DictWrapper.coerce.

        """
        if add_prefix:
            assert isinstance(x, dict)
            x = {add_prefix + k: v for k, v in x.items()}
        if isinstance(x, DictWrapper) and not kwargs:
            return x
        kwargs.setdefault('immutable', True)
        return DictWrapper.coerce(x, **kwargs)

    @property
    def dest(self):
        r"""DictWrapper: Destination dictionary for added keys."""
        for x in self.members:
            if x.mutable:
                return x
        raise ImmutableDictException("No immutable members")

    def __getitem__(self, k):
        for x in self.members:
            if k in x:
                return x[k]
        raise KeyError(k)

    def __setitem__(self, k, v):
        dest = self.dest
        for x in self.members:
            if x is dest:
                dest[k] = v
                return
            if k in x:
                raise ImmutableDictException(
                    f'Setting \"{k}\" will not change the contents as '
                    f'it is present in an immutable member proceeding '
                    f'the first mutable member'
                )

    def __delitem__(self, k):
        for x in self.members:
            if k not in x:
                continue
            if not x.mutable:
                raise ImmutableDictException(
                    f'Cannot delete \"{k}\". It is present in an '
                    f'immutable member.'
                )
            del x[k]

    def __iter__(self):
        prev = []
        for x in self.members:
            for k, v in x.items():
                if k in prev:
                    continue
                yield k
                prev.append(k)

    def _get_iterator(self, raw=False, prev=None):
        if prev is None:
            prev = []
        for x in self.members:
            for k, v in x._get_iterator(raw=raw, prev=prev):
                yield (k, v)

    @contextlib.contextmanager
    def temporary_prefix(self, prefix, per_member=False, **kwargs):
        r"""Temporarily set a prefix for the dictionary with the context.
        When the context exits, any previous prefix will be restored and
        any added parameters will be maintained.

        Args:
            prefix (str): Prefix to use within the context.
            per_member (bool, optional): If True, add prefixes on a
                member-by-member basis.
            **kwargs: Additional keyword arguments are passed to
                temporary_prefix for each member if per_member is True,
                and the parent method otherwise.

        Yields:
            DictWrapper: View of the dictionary with the provided prefix.

        """
        if not per_member:
            with super(DictSet, self).temporary_prefix(
                    prefix, **kwargs) as child:
                yield child
            return
        with contextlib.ExitStack() as stack:
            for x in self.members:
                stack.enter_context(x.temporary_prefix(prefix, **kwargs))
            yield self

    def insert(self, idx, member, **kwargs):
        r"""Add a new member to the set.

        Args:
            idx (index): Index to insert the new member add.
            member (dict): New member.
            **kwargs: Additional keyword arguments are passed to
                coerce_member.

        """
        self.members.insert(idx, self.coerce_member(member, **kwargs))

    def append(self, member, **kwargs):
        r"""Add a new member to the end of the set.

        Args:
            member (dict): New member.
            **kwargs: Additional keyword arguments are passed to
                coerce_member.

        """
        self.members.append(self.coerce_member(member, **kwargs))


# class SplitDict(DictSet):
#     r"""Dictionary where updates are made to a separate destination
#     dictionary.

#     Args:
#         src (dict): Dictionary that keys should be drawn from.
#         dst (dict): Dictionary that assignment should be performed on.
#             If not provided, assignment will be to src, but with the
#             provided dst_prefix string as a prefix. If True, a new
#             destination dictionary will be created and populated with all
#             of the keys from src. If False, assignment will not be
#             allowed.

#     """

#     def __init__(self, src, dst=None, src_prefix=None, dst_prefix=None):
#         if dst_prefix is None:
#             dst_prefix
#         src = PrefixedDict(src, prefix=src_prefix, immutable=True)
#         if dst_prefix is None and dst is not False:
#             dst_prefix = src.prefix
#         if dst is None:
#             dst = PrefixedDict(src, prefix=dst_prefix)
#         elif dst is True:
#             dst = PrefixedDict(
#                 {
#                     k: v for k, v in src._get_iterator(raw=True)
#                     if k.startswith(dst_prefix)
#                 },
#                 prefix=dst_prefix,
#             )
#         self._src = src
#         self._dst = dst
#         members = [x for x in [dst, src] if x is not False]
#         super(SplitDict, self).__init__(members)

#     @property
#     def dest(self):
#         r"""DictWrapper: Destination dictionary for added keys."""
#         if self._dst is False:
#             raise ImmutableDictException("No destination provided")
#         return self._dst


############################################################
# LPy parametrization class
#  - 'age' indicates the time since germination
#  - 'n' indicates the phytomer count
############################################################


class PlantParameterBase(RegisteredClassBase):
    r"""Base class for managing architecture parameters."""

    _registry_key = 'plant_parameter'
    _properties = {}
    _defaults = {}
    _property_dependencies = {}
    _property_dependencies_defaults = {}
    _length_parameters = [
        'Length', 'Width', 'Thickness',
    ]
    _area_parameters = [
        'Area',
    ]
    _angle_parameters = [
        'Angle',
    ]
    _unit_dimension = None

    @staticmethod
    def _add_parameter_arguments(cls, dst, name=None,
                                 existing=None, **kws):
        if existing is None:
            existing = [x[0][0] for x in dst._arguments]
        if name is None:
            name = ''
        for k, v in cls._properties.items():
            cls._add_parameter_argument(cls, dst, name, existing,
                                        k, v, **kws)

    @staticmethod
    def _add_parameter_argument(cls, dst, name, existing, k, v,
                                parents=None, dependencies=None, **kws):
        if existing is None:
            existing = [x[0][0] for x in dst._arguments]
        if name is None:
            name = ''
        kname = name if name.endswith(k) else name + k
        if isinstance(v, type):
            parents = [] if parents is None else copy.deepcopy(parents)
            parents.append((name, cls))
            dependencies = cls._get_property_dependencies(
                k, prefix=name, existing=dependencies,
            )
            v._add_parameter_arguments(v, dst, name=kname,
                                       parents=parents,
                                       existing=existing,
                                       dependencies=dependencies,
                                       **kws)
            return
        new_arg = cls._property2argument(cls, dst, name, k, v,
                                         parents=parents,
                                         dependencies=dependencies,
                                         **kws)
        if new_arg[0][0] not in existing:
            dst._arguments.append(new_arg)
            existing.append(new_arg[0][0])

    @staticmethod
    def _property2argument(cls, dst, name, k, json,
                           dependencies=None, parents=None, **kws):
        kname = name if name.endswith(k) else name + k
        kwargs = jsonschema2argument(json)
        if kname in dst._defaults:
            kwargs['default'] = dst._defaults[kname]
        if 'help' not in kwargs and cls._help:
            kwargs['help'] = cls._help
        kwargs.update(**kws)
        dependencies = cls._get_property_dependencies(
            k, prefix=name, existing=dependencies,
        )
        if 'help' in kwargs:
            while '{' in kwargs['help']:
                kwargs['help'] = cls._format_help(
                    cls, dst, kwargs['help'], name, k, parents=parents
                )
            if dependencies[-1]:
                provided = []
                not_provided = []
                conditions = []
                for kdep, vdep in dependencies[-1].items():
                    if vdep is True:
                        provided.append(kdep)
                    elif vdep is False:
                        not_provided.append(kdep)
                    else:
                        assert isinstance(vdep, list)
                        conditions.append(f'{kdep} one of {vdep}')
                if provided:
                    verb = 'is' if len(provided) == 1 else 'are'
                    conditions.append(f'{provided} {verb} provided')
                if not_provided:
                    verb = 'is' if len(not_provided) == 1 else 'are'
                    conditions.append(f'{not_provided} {verb} not provided')
                conditions = format_list_for_help(conditions)
                kwargs['help'] += f' [Only valid if {conditions}]'
        if any(dependencies):
            kwargs['dependencies'] = dependencies
        return ((f'--{kname}', ), kwargs)

    @staticmethod
    def _format_help(cls, dst, msg, name, k, parents=None):
        kname = name if name.endswith(k) else name + k
        fmtkws = {
            'class_name': cls._name,
            'class_help': cls._help,
            'parent': name,
            'Component': dst._property2component(kname),
        }
        if parents:
            fmtkws['parents'] = [x[0] for x in parents[::-1]]
        if cls._name is not None:
            for k, v in list(fmtkws.items()):
                if v is None:
                    fmtkws.pop(k)
        if isinstance(fmtkws.get('Component', None), str):
            fmtkws['component'] = fmtkws['Component'].lower()
        try:
            return msg.format(**fmtkws)
        except KeyError:
            print(cls, kname)
            raise

    @classmethod
    def _get_required_properties(cls, name=None):
        if name is None:
            out = []
            for k in cls._required:
                out += cls._get_required_properties(k)
            return out
        if isinstance(cls._properties[name], type):
            return [f'{name}{k}' for k in
                    cls._properties[name]._get_required_properties()]
        return [name]

    @classmethod
    def _get_property_dependencies(cls, name, prefix='', existing=None,
                                   base=None):
        out = [] if existing is None else copy.deepcopy(existing)
        out.append({} if base is None else copy.deepcopy(base))
        if name not in cls._property_dependencies:
            return out
        for k, v in cls._property_dependencies[name].items():
            reqprops = cls._get_required_properties(k)
            for x in reqprops:
                subx = x if prefix is None else f'{prefix}{x}'
                out[-1][subx] = v
        return out


def DelayedPlantParameter(name, **kwargs):
    r"""Class factory for creating a plant parameter class on
    registration.

    Args:
        name (str): Name of the delayed plant parameter class.
        **kwargs: Additional keyword arguments are passed to the
            specialize method for the class.

    Returns:
       type: Dummy plant parameter class that will access a registered
           plant parameter class when it is used to create an instance.

    """

    spec_name = kwargs.pop('specialize_name', None)

    def parameter_class():
        cls = get_class_registry().get('plant_parameter', name)
        if spec_name is not None:
            cls = cls.specialize(spec_name, **kwargs)
            # TODO: Prevent duplication?
        return cls

    class DelayedPlantParameterMeta(type):

        def __call__(self, *args, **kwargs):
            return parameter_class()(*args, **kwargs)

        def __getattr__(self, k):
            return getattr(parameter_class(), k)

    class _DelayedPlantParameter(object, metaclass=DelayedPlantParameterMeta):
        pass

    return _DelayedPlantParameter


class ParameterDict(DictSet):
    r"""Class for managing a set of parameters.

    Args:
        provided (dict, optional): Set of user provided parameters.

    """

    def __init__(self, provided=None, **kwargs):
        self.provided = PrefixedDict(provided, immutable=True, **kwargs)
        self.previous = PrefixedDict(DictSet([]), immutable=True, **kwargs)
        self.defaults = PrefixedDict(DictSet([]), immutable=True, **kwargs)
        self.component = None
        self._dest = PrefixedDict({}, **kwargs)
        members = [
            self.provided, self.previous, self._dest, self.defaults,
        ]
        super(ParameterDict, self).__init__(members, **kwargs)

    @property
    def dest(self):
        r"""DictWrapper: Destination dictionary for added keys."""
        return self._dest

    @property
    def prefix(self):
        r"""str: Current prefix."""
        return self.dest.prefix

    @contextlib.contextmanager
    def temporary_source_prefix(self, prefix, **kwargs):
        r"""Temporarily create a context where the source dictionaries
        have a different prefix than the destination dictionary.

        Args:
            prefix (str): Temporary prefix to set for the sources.
            **kwargs: Additional keyword arguments are passed to
                temporary_prefix for each source dictionary.

        Yields:
            ParameterDict: Self with temporary prefix applied.

        """
        with contextlib.ExitStack() as stack:
            for x in [self.provided, self.previous, self.defaults]:
                stack.enter_context(x.temporary_prefix(prefix, **kwargs))
            yield self

    @contextlib.contextmanager
    def updating(self, cls):
        r"""Context manager for updating the parameters for a class.

        Args:
            cls (PlantParameterBase): Parameter instance that is being
                updated. Default parameters will be added to the current
                set based on the existing parameters and the class's
                prefix will be added to the dictionary prefix within the
                context.

        Yields:
            ParameterDict: Self with temporary prefix applied.

        """
        defaults0 = self.defaults.members
        previous0 = self.previous.members
        added_aliases = {}
        is_component = (not self.prefix)
        if is_component and cls.name:
            assert self.component is None
            self.component = cls.name
        with self.temporary_prefix(cls.name, append=True,
                                   per_member=True):
            try:
                for k, v in cls._aliases.items():
                    if k in self.provided:
                        self[v] = self.provided[k]
                        added_aliases[v] = k
                if cls._defaults and not cls.no_class_defaults:
                    self.defaults.append(cls._defaults, add_prefix=True,
                                         logger=self.logger)
                if getattr(cls, 'parameters', None):
                    self.previous.append(cls.parameters, add_prefix=True,
                                         logger=self.logger)
                self.copy_external_properties(cls)
                self.copy_component_properties(cls)
                self.add_property_dependency_defaults(cls)
                cls.add_class_defaults(self)
                yield self
            finally:
                self.defaults.members = defaults0
                self.previous.members = previous0
                if cls.initialized:
                    with self.provided.report_change(
                            prefix='Removing local: '):
                        try:
                            self.provided._immutable = False
                            for k in cls.parameters.keys():
                                if k in self.provided:
                                    self.provided.pop(k)
                                if k in added_aliases:
                                    self.provided.pop(added_aliases[k])
                        finally:
                            self.provided._immutable = True
                if is_component:
                    self.component = None

    def add_property_dependency_defaults(self, cls):
        r"""Add defaults based on the other properties that are present
        and the _property_dependencies_defaults attribute of class.

        Args:
            cls (PlantParameterBase): Parameter instance that is being
                updated.

        """
        if not cls._property_dependencies_defaults:
            return
        with self.report_change(prefix='Property dependency defaults: '):
            for k, v in cls._property_dependencies_defaults.items():
                if self.get(k, NoDefault) is not NoDefault:
                    continue
                for dep, cond in v['conditions'].items():
                    if cls._eval_cond(self, dep, cond):
                        self[k] = v['value']

    def copy_external_properties(self, cls):
        r"""Copy external properties identified by cls.

        Args:
            cls (PlantParameterBase): Parameter instance that is being
                updated.

        """
        if not cls._external_properties:
            return
        with self.temporary_source_prefix(
                '', report_change='External properties: '):
            for k in cls._external_properties:
                self.copy_src2dst(k)

    def copy_component_properties(self, cls):
        r"""Copy component properties identified by cls.

        Args:
            cls (PlantParameterBase): Parameter instance that is being
                updated.

        """
        component = self.component
        if not (isinstance(component, str) and cls._component_properties):
            return
        with self.temporary_source_prefix(
                '', report_change='Component properties: '):
            for k in cls._component_properties:
                self.copy_src2dst(k, f'{component}{k}')

    @property
    def local_parameters(self):
        r"""dict: Parameters with the current prefix."""
        # TODO: Remove keys based on property dependencies?
        out = self.select_prefix(self.prefix, strip=True)
        return out


class SimplePlantParameter(PlantParameterBase):
    r"""Class for simple parameters requiring a single value.

    Args:
        name (str, optional): Parameter name without prefix.
        param (dict, optional): Parameters (specified via full parameter
            names with prefixes) that this parameter should take
            properties from. If not provided and parent is not provided,
            kwargs will be used.
        parent (PlantParameterBase, optional): Parent parameter that
            uses this parameter.
        defaults (dict, optional): Defaults (with only this parameter's
            prefixes) that should be used for parameters missing from
            param.
        seed (int, optional): Seed that should be used for the random
            number generator to make geometries reproducible.
        verbose (bool, optional): If True, log messages will be printed
            describing the generation process.
        debug (bool, optional): If True, pdb break points will be set
            for debug messages and log messages will be displayed.
        debug_param (list, optional): Parameters (expressed as full names)
            that debug mode should be turned on for (only exact parameter
            matches are debugged, not children).
        debug_param_prefix (list, optional): Prefix of parameters that
            debug mode should be turned on for.
        required (bool, optional): If True, this parameter is required
            and errors will be raised if it is not fully specified.
        **kwargs: Additional keyword arguments are only allowed for
            root parameters (that don't have a parent) in which case
            they are treated as param.

    Attributes:
        parameters (dict): Child parameters used by this parameter.

    ClassAttributes:
        _name (str): Name that should be used to register the class and
            be used as the default prefix.
        _properties (dict): Set of properties that control the parameter.
        _defaults (dict): Set of defaults for properties controling the
            parameter.
        _aliases (dict): Set of aliases mapping between alias names for
            parameters and the parameters they map to.
        _required (list): Set of properties that are required by this
            parameter.
        _variables (list): Properties that reference other variables.
        _dependencies (list): Root level properties that this parameter
            uses.
        _subschema_keys (list): JSON schema properties that contain
            schemas.
        _component_properties (list): Properties that can be defined for
            all parameters associated with a component.
        _attribute_properties (list): Properties that should be
            accessible directly as class attributes.
        _index_var (list): Variables that will be supplied as input to
            calls to generate parameters.
        _components (list): Plant components that may be used with
            _component_properties.

    """

    _name = 'simple'
    _help = None
    _properties = {
        '': {
            'type': ['string', 'null', 'number', 'boolean']
        },
    }
    _aliases = {}
    _required = []
    _variables = []
    _specialized = []
    _dependencies = []
    _subschema_keys = ['oneOf', 'allOf']
    _external_properties = []
    _component_properties = []
    _attribute_properties = []
    _index_var = ['X', 'N', 'Age']
    _components = {}

    def __init__(self, name, param, parent, required=False):
        super(SimplePlantParameter, self).__init__()
        assert isinstance(param, ParameterDict)
        self._key_stack = []
        self._cache = {}
        self._generators = {}
        self.name = name
        self.parameters = {}
        self.parent = parent
        self.required = required
        self.initialized = False
        self.child_parameters = self.parameter_names(self.fullname, 'children')
        self.core_paremeters = self.parameter_names(self.fullname, 'core')
        self.valid_parameters = self.child_parameters + self.core_paremeters
        self.update(param)

    def clear(self):
        r"""Clear the parameter class."""
        self.initialized = False
        self.parameters.clear()
        self.defaults.clear()

    def update(self, param):
        r"""Update the parameters.

        Args:
            param (dict): New parameters.

        """
        in_init = (len(self.parameters) == 0)
        if not isinstance(param, ParameterDict):
            param = ParameterDict(param)
        param0 = param.provided.flattened
        keys0 = list(param0.keys())
        schema = self.schema()
        self.initialization_error = None
        with param.updating(self):
            self.parameters = {}
            self.initialized = self._parse_single(param, schema)
        if self.initialized:
            self.log(f'Parsed param:\n{pprint.pformat(self.parameters)}')
        elif self.required:
            self.error(rapidjson.NormalizationError,
                       f'Failed to initialize required parameter:\n'
                       f'{self.initialization_error}')
        else:
            self.log('MISSING PARAMETERS')
        if self.initialized:
            self.log(f'BEFORE:\n{pprint.pformat(param0)}')
            self.log(f'SCHEMA:\n{pprint.pformat(schema)}')
            self.log(f'AFTER:\n{pprint.pformat(param.provided)}')
        else:
            DictWrapper.assert_keys_match(param.provided, keys0,
                                          logger=self)
        if self.parent is None and param.provided:
            self.debugging = True
            self.error(AssertionError,
                       f'Unparsed param:\n{pprint.pformat(param.provided)}')
        if self.initialized:
            msg = f'INITIALIZED:\n{pprint.pformat(self.contents)}'
            debug = ((not in_init) or self.parent == self.root)
            self.log(msg, debug=debug)

    @classmethod
    def _get_default(cls, name, param=None):
        if name in cls._defaults:
            return cls._defaults[name]
        property_entry = cls._properties[name]
        if isinstance(property_entry, type):
            if '' in property_entry._properties:
                return property_entry._get_default('')
            elif param is not None:
                with temporary_prefix(param, name, append=True):
                    if all(property_entry.is_enabled(kreq, param)
                           for kreq in property_entry._required):
                        return True
        elif 'default' in property_entry:
            return property_entry['default']
        return NoDefault

    @classmethod
    def _eval_cond(cls, param, k, cond, prefix='', value=None,
                   include_defaults=False):
        k = f'{prefix}{k}'
        default = NoDefault
        if include_defaults:
            default = cls._get_default(
                k, param=(None if isinstance(cond, list) else param),
            )
        if value is None:
            value = (
                param.get(k, default)
                if isinstance(param, (dict, DictWrapper))
                else getattr(param, k, default)
            )
        return (((cond is True and value is not NoDefault)
                 or (cond is False and value is NoDefault)
                 or (isinstance(cond, (list, tuple))
                     and value not in cond)))

    @classmethod
    def is_enabled(cls, name, param, prefix='', include_defaults=False):
        r"""Check if a parameter is enabled based on other provided
        parameters.

        Args:
            name (str): Parameter name.
            param (dict, object): Object to retrieve parameter values
                from.
            prefix (str, optional): Prefix for parameter names.
            include_defaults (bool, optional): Consider defaults when
                 determining if a parameter is enabled.

        Returns:
            bool: True if the parameter is enabled, False otherwise.

        """
        assert name in cls._properties
        if name not in cls._property_dependencies:
            return True
        for k, v in cls._property_dependencies[name].items():
            if not cls._eval_cond(param, k, v, prefix=prefix,
                                  include_defaults=include_defaults):
                return False
        return True

    @classmethod
    def add_class_defaults(cls, param, **kwargs):
        r"""Update defaults based on parameters.

        Args:
            param (ParameterDict): Parameter set to update.
            **kwargs: Additional keyword arguments are added to param.

        """
        assert isinstance(param, ParameterDict)
        with param.report_change(prefix='kwargs: '):
            param.update(kwargs)

    @classmethod
    def schema_contains(cls, schema, key, value=None):
        r"""Check if a schema contains a property corresponding to the
        provided key.

        Args:
            schema (dict): Schema to check.
            key (str): Property to check for.
            value (object, optional): Value associated with the key.

        Returns:
            bool: True if schema contains a property for key.

        """
        if key in schema.get('properties', {}):
            return True
        if key in schema.get('required', []):
            return True
        for ksub in cls._subschema_keys:
            for v in schema.get(ksub, []):
                if cls.schema_contains(v, key, value=value):
                    return True
        return False

    @classmethod
    def specialize(cls, name, exclude=[], **kwargs):
        r"""Created a specialized version of this class.

        Args:
            name (str): Name of the root property for the class.
            exclude (list, optional): Properties that should be excluded.
            properties (dict, optional): Properties that should be added.
            defaults (dict, optional): Defaults that should be added.
            property_dependencies (dict, optional): Property dependencies
                that should be added.
            **kwargs: Additional keyword arguments are added as attributes
                to the resulting class.

        """
        exclude = exclude + cls._specialized
        class_dict = copy.deepcopy(kwargs)
        class_dict.setdefault('_name', name)
        update_attr = [
            'properties', 'required', 'defaults',
            'property_dependencies',
            'property_dependencies_defaults',
            'component_properties',
            'external_properties',
        ]
        for k in update_attr:
            kattr = f'_{k}'
            vattr = getattr(cls, kattr)
            if isinstance(vattr, list):
                class_dict.setdefault(
                    kattr, [
                        k for k in vattr if k not in exclude
                    ]
                )
            elif isinstance(vattr, dict):
                class_dict.setdefault(
                    kattr, {
                        k: copy.deepcopy(v) for k, v in vattr.items()
                        if k not in exclude
                    }
                )
            if k in ['property_dependencies_defaults'] and exclude:
                for v in class_dict[kattr].values():
                    v['conditions'] = {
                        kk: vv for kk, vv in v['conditions'].items()
                        if kk not in exclude
                    }
            v = class_dict.pop(k, None)
            if not v:
                continue
            if isinstance(v, list):
                assert isinstance(class_dict[kattr], list)
                for x in v:
                    if x not in class_dict[kattr]:
                        class_dict[kattr].append(x)
            elif isinstance(v, dict):
                assert isinstance(class_dict[kattr], dict)
                class_dict[kattr].update(**v)
            else:
                raise TypeError(type(v))
        created = RegisteredMetaClass(
            f'Created{name}', (cls,), class_dict)
        return created

    @property
    def prefix(self):
        r"""str: Prefix that child parameters will have."""
        return self.name

    def remove_prefix(self, k, use_fullname=False):
        r"""Remove the current parameter prefix from a string if it
        matches.

        Args:
            k (str): String to remove prefix from.
            use_fullname (bool, optional): If True, the full name should
                be used as the prefix.

        Returns:
            str: String with the prefix removed.

        Raises:
            ValueError: If k does not start with the prefix.

        """
        if use_fullname:
            prefix = self.fullname
        else:
            prefix = self.prefix
        if not prefix:
            return k
        if k.startswith(prefix):
            return k.split(prefix, 1)[-1]
        raise ValueError(f"String \"{k}\" does not start with the "
                         f"prefix \"{prefix}\"")

    def prefixes(self, k, use_fullname=False):
        r"""Check if a string starts with the prefix for this parameter.

        Args:
            k (str): String to check.
            use_fullname (bool, optional): If True, the full name should
                be used as the prefix.

        Returns:
            bool: True if k starts with this parameter's prefix.

        """
        if use_fullname:
            prefix = self.fullname
        else:
            prefix = self.prefix
        if not prefix:
            return True
        return k.startswith(prefix)

    @property
    def fullname(self):
        r"""str: Full name with parent prefix."""
        out = self.prefix
        if self.parent is not None:
            out = f'{self.parent.fullname}{out}'
        return out

    @property
    def component(self):
        r"""PlantComponent: Component that this property belongs to."""
        if self.parent:
            return self.parent.component
        return None

    @property
    def component_name(self):
        r"""str: Name of the component that this property belongs to."""
        component = self.component
        if component is not None:
            return component.name
        return None

    @property
    def root(self):
        r"""PlantParameterBase: Root parameter."""
        if self.parent is not None:
            return self.parent.root
        return self

    @property
    def verbose(self):
        r"""bool: If True, log messages will be emitted."""
        if self.debugging:
            return True
        if self.parent is not None:
            return self.root._verbose
        return self._verbose

    @property
    def no_class_defaults(self):
        if self.parent is not None:
            return self.root._no_class_defaults
        return self._no_class_defaults

    @property
    def debug_param(self):
        r"""list: Parameters that debug mode should be enabled for."""
        if self.parent is not None:
            return self.root._debug_param
        return self._debug_param

    @property
    def debug_param_prefix(self):
        r"""list: Parameters that debug mode should be enabled for."""
        if self.parent is not None:
            return self.root._debug_param_prefix
        return self._debug_param_prefix

    @property
    def debugging(self):
        r"""bool: True if debugging is active."""
        if self.fullname.startswith(tuple(self.debug_param_prefix)):
            return True
        if self.fullname in self.debug_param:
            return True
        if self.parent is not None:
            return self.root._debugging
        return self._debugging

    @debugging.setter
    def debugging(self, value):
        if self.parent:
            self.root.debugging = value
            return
        self._debugging = value
        if value:
            self.log('Turning on debugging')

    def debug(self, message='', force=False, **kwargs):
        r"""Set a pdb break point if debugging is active.

        Args:
            message (str, optional): Log message to show before setting
                break point.
            force (bool, optional): If True, the break point will be
                set even if debugging is not active.
            **kwargs: Additional keyword arguments are passed to the
                log method if it is called.

        """
        self.log(f'DEBUG: {message}', force=force, **kwargs)
        if not (force or self.debugging):
            return
        pdb.set_trace()

    # Methods for generating parameters
    def hasmethod(self, k):
        r"""Check if there is an explicit method for a parameter.

        Args:
            k (str): Parameter name.

        Returns:
            bool: True if there is an explicit method, False otherwise.

        """
        return hasattr(type(self), k)

    def __getattr__(self, k):
        if self.hasmethod(k):
            raise AttributeError(f'{self}: {k}')
        if self.parent is not None:
            self.error(AttributeError, f'{self}: {k}')
        if k in self._attribute_properties:
            return self.parameters.get(k, None)
        if k not in self.valid_parameters:
            pprint.pprint(self.valid_parameters)
            self.error(AttributeError, f'{self}: {k}')
        return functools.partial(self.getfull, k, k0=k)

    def getfull(self, k, age=None, n=None, x=None, **kwargs):
        r"""Get a parameter value with the index fully specified as
        arguments.

        Args:
            k (str): Name of parameter to return without any prefixes.
            age (float): Age of the component that the parameter should
                be returned for.
            n (int):  Phytomer count of the component that the parameter
                should be returned for.
            x (float, optional): Position along the component that the
                parameter should be returned for.
            **kwargs: Additional keyword arguments are passed to get.

        Returns:
            object: Parameter value.

        """
        idx = (age, n, x)
        return self.get(k, idx=idx, **kwargs)

    def set(self, k, value, k0=None):
        r"""Set a parameter value.

        Args:
            k (str): Name of parameter to set, without any prefixes.
            value (object): Value to set parameter to.
            k0 (str): Full name of the parameter including any parent
                prefixes. If not provided, the full name will be
                created based on the current prefix.

        """
        raise NotImplementedError

    @property
    def children(self):
        r"""iterable: Child parameter instances."""
        for v in self.parameters.values():
            if isinstance(v, PlantParameterBase) and v.initialized:
                yield v

    @property
    def dependencies(self):
        r"""list: Set of variables that this parameter is dependent on."""
        out = copy.deepcopy(self._dependencies)
        out += [self.parameters.get(k, None) for k in self._variables
                if k in self.parameters]
        for child in self.children:
            out += child.dependencies
        return list(set(out))

    @property
    def all_parameters(self):
        r"""dict: Set of all parameters, including children."""
        out = PrefixedDict({}, prefix=self.prefix)
        for k, v in self.parameters.items():
            if isinstance(v, PlantParameterBase):
                assert v.initialized
                out.update(**v.all_parameters)
            else:
                out[k] = v
        return out.raw_flattened

    @property
    def constant_var(self):
        r"""list: Set of index variables that this parameter is
        independent of."""
        dependencies = self.dependencies
        return [k for k in self._index_var if k not in dependencies]

    def _nullify_constant_var(self, idx, var=None, context=None):
        if var is None:
            var = self.constant_var
        if isinstance(var, list):
            for v in var:
                idx = self._nullify_constant_var(idx, var=v,
                                                 context=context)
            return idx
        elif var is None:
            return idx
        age, n, x = idx[:]
        if var == 'Age':
            age = None
        elif var == 'X':
            x = None
        elif var == 'N':
            n = None
        else:
            raise ValueError(f"Unsupported constant var \"{var}\"")
        return (age, n, x)

    @classmethod
    def get_class(cls, k, prefix=''):
        r"""Get a parameter class.

        Args:
            k (str): Name of parameter to return, without any prefixes.
            k0 (str, optional): Full name of the parameter including any
                parent prefixes. If not provided, the full name will be
                created based on the current prefix.

        Returns:
            PlantParameterBase: Parameter class.

        """
        if prefix and not k.startswith(prefix):
            raise KeyError(f'\'{k}\' does not start with prefix '
                           f'\'{prefix}\'')
        if prefix:
            k = k.split(prefix, 1)[-1]
        if k in cls._properties:
            return cls._properties[k]
        for kk, vv in cls._properties.items():
            if isinstance(vv, dict):
                continue
            try:
                return vv.get_class(k, prefix=kk)
            except KeyError:
                pass
        raise KeyError(k)

    def get(self, k, default=NoDefault, idx=None, cache_key=None,
            k0=None, return_other=None, **kwargs):
        r"""Get a parameter value.

        Args:
            k (str): Name of parameter to return, without any prefixes.
            default (object, optional): Default to return if the
                parameter is not set.
            idx (tuple, optional): Simulation index to get the parameter
                value for. If not provided, the current index will be
                used based on the last parameter accessed with an index.
            cache_key (tuple, optional): Key to use to cache any
                generated values if different than the default.
            k0 (str, optional): Full name of the parameter including any
                parent prefixes. If not provided, the full name will be
                created based on the current prefix.
            return_other (str, optional): Name of what should be
                returned instead of the parameter value.
            **kwargs: Additional keyword arguments are passed to any
                nested calls to get or generate.

        Returns:
            object: Parameter value.

        """
        if idx is None:
            idx = self._current_idx(cache_key=cache_key)
        kidx = self._nullify_constant_var(idx, context=k)
        age, n, x = kidx[:]
        if k0 is None:
            is_child = True
            k0 = f'{self.fullname}{k}'
        else:
            is_child = self.prefixes(k)
            if is_child:
                k = self.remove_prefix(k)
        if k == 'Age':
            assert age is not None
            return age
        elif k == 'N':
            assert n is not None
            return n
        elif k == 'X':
            assert x is not None
            return x
        elif k0 not in self.valid_parameters:
            if is_child:
                if default is not NoDefault:
                    return default
                self.error(KeyError,
                           f'Missing child parameter \"{k}\" '
                           f'(k0=\"{k0}\")?', debug=True)
            self.log(f'External parameter \"{k0}\"')
            return self.root.get(k0, default=default, idx=idx,
                                 return_other=return_other, **kwargs)
        if k and k not in self.parameters:
            for kk, vv in self.parameters.items():
                if isinstance(vv, PlantParameterBase) and vv.prefixes(k):
                    return vv.get(k, idx=kidx, k0=k0,
                                  cache_key=cache_key, default=default,
                                  return_other=return_other, **kwargs)
            else:
                if default is not NoDefault:
                    return default
                self.error(KeyError, f"Unsupported var \"{k0}\"")
        if not return_other:
            out = self._push_key(k0, *kidx, cache_key=cache_key)
            if out is not None:
                return out
        try:
            if k == '':
                v = self
            elif k in self.parameters:
                v = self.parameters[k]
            else:
                self.error(KeyError, f"Unsupported var? \"{k0}\" (\"{k}\")")
            if isinstance(v, PlantParameterBase):
                if not v.initialized:
                    self.error(KeyError, f"Parameter not initialized \"{k0}\"")
                if return_other is None:
                    self.log(f'Generating \"{k0}\"')
                    v = v.generate(kidx, **kwargs)
                elif return_other == 'parameters':
                    v = v.parameters
                elif return_other == 'instance':
                    pass
                else:
                    raise ValueError(f'Invalid return_other = '
                                     f'\"{return_other}\"')
        except BaseException:
            if default is NoDefault:
                raise
            v = default
        if not return_other:
            self._pop_key(v)
        return v

    def generate(self, idx, **kwargs):
        r"""Generate this parameter.

        Args:
            idx (tuple): Index variables that should be used to generate
                this parameter.
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        out = self.parameters['']
        self.log(f'{self.fullname} = {out}')
        return out

    @classmethod
    def schema(cls):
        r"""Create a JSON schema for parsing parameters used by this
        parameter.

        Returns:
            dict: JSON schema.

        """
        out = {
            'type': 'object',
            'properties': copy.deepcopy(cls._properties),
        }
        if cls._required:
            out['required'] = cls._required.copy()
        return out

    @classmethod
    def parameter_names(cls, name, scope='all'):
        r"""Get a list of valid parameter names.

        Args:
            name (str): Base name for parameters.
            scope (str, optional): Scope for parameters that should
                be returned. Options are:
                  'all': Return all parameters.
                  'core': Return parameters only directly allowed by
                      this parameter.
                  'children': Return parameters only indirectly allowed
                      by child parameters.

        Returns:
            list: Parameter names.

        """
        out = []
        if scope in ['all', 'core']:
            out += [f'{name}{k}' for k in cls._properties.keys()]
            if name not in out:
                out.insert(0, name)
        if scope in ['all', 'children']:
            for k, v in cls._properties.items():
                if isinstance(v, type):
                    kname = f'{name}{k}'
                    out += v.parameter_names(kname)
        return out

    def _extract_parameters(self, schema):
        parameters = {'properties': {}, 'required': []}
        for k in list(schema.get('properties', {}).keys()):
            if isinstance(schema['properties'][k], type):
                parameters['properties'][k] = schema['properties'].pop(k)
                if k in schema.get('required', []):
                    parameters['required'].append(k)
                    schema['required'].remove(k)
        if not schema.get('properties', {}):
            schema.pop('properties', None)
        if not schema.get('required', []):
            schema.pop('required', None)
        if not parameters['properties']:
            parameters.pop('properties')
        if not parameters['required']:
            parameters.pop('required')
        return parameters

    def _parse_single(self, param, schema, child_param=None,
                      parameters=None, required=None, idstr='',
                      subschema_errors=None):
        if idstr:
            self.log(f'{idstr}: {schema}')
        if child_param is None:
            child_param = param.local_parameters
            self.log(f'CHILDREN:\n{pprint.pformat(child_param)}')
        if required is None:
            required = self.required
        # Parse subschemas w/ parameter classes via recursion
        try:
            for k in self._subschema_keys:
                if k not in schema:
                    continue
                krequired = (k == 'allOf')
                if k == 'oneOf':
                    keys_all = set()
                    for x in schema[k]:
                        keys_all
                kparam = [
                    self._extract_parameters(x) for x in schema[k]
                ]
                if not any(kparam):
                    continue
                errors = []
                results = [
                    self._parse_single(
                        param, x, child_param=child_param,
                        idstr=f' {k}[{i}]', required=krequired,
                        parameters=kparam[i], subschema_errors=errors,
                    )
                    for i, x in enumerate(schema[k])
                ]
                if k == 'oneOf':
                    if sum(results) > 1:
                        valid = [
                            schema[k][i] for i, v in
                            enumerate(results) if v
                        ]
                        raise rapidjson.NormalizationError(
                            f'More than one match ({len(valid)}):\n'
                            f'{pprint.pformat(valid)}\n'
                            f'{pprint.pformat(child_param)}'
                        )
                    elif sum(results) == 0:
                        raise rapidjson.NormalizationError(
                            f'No matches\n{"\n".join(errors)}')
                elif k == 'allOf':
                    if sum(results) != len(results):
                        invalid = [
                            schema[k][i] for i, v in
                            enumerate(results) if not v
                        ]
                        raise rapidjson.NormalizationError(
                            f'Not all match:\n'
                            f'{pprint.pformat(invalid)}\n'
                            f'{pprint.pformat(child_param)}:\n'
                            f'{"\n".join(errors)}'
                        )
                else:
                    self.error(NotImplementedError,
                               f'Unsupported schema key {k}')
                schema.pop(k)
            # Split schema & param into regular parameters and parameter
            #   classes
            if parameters is None:
                parameters = self._extract_parameters(schema)
            child_param_schema = {
                k: v for k, v in child_param.items()
                if self.schema_contains(schema, k, value=v)
            }
            # Parse regular schema parameters
            if schema:
                try:
                    out = rapidjson.normalize(child_param_schema, schema)
                    self.log(f"NORMALIZE: {child_param_schema}\n{out}")
                except rapidjson.NormalizationError as e:
                    raise rapidjson.NormalizationError(
                        f'Normalization error: {e.args[0]}\n'
                        f'Schema param:\n'
                        f'{pprint.pformat(child_param_schema)}'
                    )
            # Parse parameter classes
            missing = []
            missing_errors = []
            for k, v in parameters.get('properties', {}).items():
                self.log(f'Adding{idstr} child parameter \"{k}\": {v}')
                krequired = (k in parameters.get('required', []))
                if isinstance(child_param.get(k, None),
                              PlantParameterBase):
                    out[k] = child_param.pop(k)
                    out[k].update(param)
                else:
                    out[k] = v(
                        k, param, self, required=(required and krequired),
                    )
                if not out[k].initialized:
                    if krequired:
                        missing.append(k)
                        missing_errors.append(out[k].initialization_error)
                    out.pop(k)
            if missing:
                raise rapidjson.NormalizationError(
                    f'Failed to initialized child parameter '
                    f'instance(s): {missing}\n'
                    f'{"\n".join(missing_errors)}'
                )
            # Update parameters on the class
            self.log(f"ADDING:\n{pprint.pformat(out)}")
            self.parameters.update(**out)
            return True
        except rapidjson.NormalizationError as e:
            if required:
                self.error(rapidjson.NormalizationError, e.args[0])
            self.log(f'Ignored error: {e.args[0]}')
            if subschema_errors is not None:
                subschema_errors.append(e.args[0])
            else:
                self.initialization_error = e.args[0]
            return False

    # Logging methods
    @property
    def contents(self):
        r"""dict: The contents of parameters."""
        param = {}
        for k, v in self.parameters.items():
            if isinstance(v, PlantParameterBase):
                if v.initialized:
                    param[k] = v.contents
            else:
                param[k] = v
        return param

    @property
    def log_prefix_instance(self):
        r"""str: Prefix to use for log messages emitted by this instance."""
        out = ''
        if self.parent:
            out = f'{self.parent.log_prefix_instance}->'
        if self.prefix:
            out += self.prefix
        else:
            out += self._name
        return out

    @property
    def log_prefix_stack(self):
        r"""str: Prefix to use for log messages describing the current
        stack."""
        prefix = ''
        key = self._current_key
        if key is not None:
            k, age, n, x = key[:]
            if x is None:
                x_str = ''
            else:
                x_str = f',x={x}'
            prefix = f'{k}[{age},{n}{x_str}]: '
        return prefix

    def log(self, message='', force=False, debug=False):
        r"""Emit a log message.

        Args:
            message (str, optional): Log message.
            force (bool, optional): If True, print the log message even
                if self.verbose is False.
            debug (bool, optional): If True, set a debug break point if
                debugging enabled.

        """
        return super(SimplePlantParameter, self).log(
            message=message, force=force,
            debug=(debug and self.debugging),
            prefix=f'{self.log_prefix_instance}: {self.log_prefix_stack}'
        )

    def error(self, error_cls, message='', debug=False):
        r"""Raise an error, adding context to the message.

        Args:
            error_cls (type): Error class.
            message (str, optional): Error message.
            debug (bool, optional): If True, set a debug break point.

        """
        return super(SimplePlantParameter, self).error(
            error_cls, message=message,
            debug=(debug or self.debugging),
            prefix=f'{self.log_prefix_instance}: {self.log_prefix_stack}'
        )

    # Methods for tracking things
    def _current_idx(self, cache_key=None):
        current_key = cache_key
        if current_key is None:
            current_key = self._current_key
        if current_key is None:
            current_key = (None, None, None, None)
        idx = current_key[1:]
        return idx

    def _push_key(self, param, age, n, x, cache_key=None):
        if self.parent:
            return self.parent._push_key(param, age, n, x,
                                         cache_key=cache_key)
        if cache_key is None:
            key = (param, age, n, x)
        else:
            key = cache_key
        if key in self._cache:
            # self.log(f"Using cached ({key})")
            return self._cache[key]
        self._key_stack.append(key)

    def _pop_key(self, out=None):
        if self.parent:
            return self.parent._pop_key(out=out)
        if out is not None:
            self._cache[self._current_key] = out
        self.log(str(out))
        self._key_stack.pop()

    @property
    def _current_key(self):
        if self.parent:
            return self.parent._current_key
        if self._key_stack:
            return self._key_stack[-1]
        return None

    @property
    def generator(self):
        r"""np.random.Generator: Current random number generator."""
        if self.parent:
            return self.parent.generator
        seed = self.seed
        if self._current_key and (self._current_key[2] is not None):
            seed += self._current_key[2]
        if seed not in self._generators:
            self.log(f"Creating generator for seed = {seed}")
            self._generators[seed] = np.random.default_rng(
                seed=seed)
        return self._generators[seed]

    def sample_dist(self, profile, *args, **kwargs):
        r"""Sample a distribution using the current random number
        generator.

        Args:
            profile (str): Name of the profile that should be sampled.
            *args, **kwargs: Additional arguments are passed to the
                generator method for the specified profile.

        """
        if not args:
            args = None
        return DistributionPlantParameter.sample_generator_dist(
            self.generator, profile=profile, args=args, **kwargs)


class OptionPlantParameter(SimplePlantParameter):
    r"""Class for parameter with exclusive options."""

    _name = None
    _name_option = ''
    _properties = {}
    _option_dependencies = {}
    _default = NoDefault

    @staticmethod
    def _on_registration(cls):
        opt = cls._name_option
        cls._properties = copy.deepcopy(cls._properties)
        cls._property_dependencies = copy.deepcopy(
            cls._property_dependencies)
        choices = list(sorted(cls._option_dependencies.keys()))
        if opt is not None and choices:
            if opt not in cls._properties:
                cls._properties[opt] = {}
            if opt not in cls._required:
                cls._required = copy.deepcopy(cls._required)
                cls._required.insert(0, opt)
            cls._properties[opt].setdefault('enum', choices)
            for k, v in cls._option_dependencies.items():
                for vv in v:
                    cls._property_dependencies.setdefault(vv, {})
                    cls._property_dependencies[vv].setdefault(opt, [])
                    cls._property_dependencies[vv][opt].append(k)
            for k in cls._properties.keys():
                if k == opt:
                    continue
                cls._property_dependencies.setdefault(k, {})
                cls._property_dependencies[k].setdefault(opt, True)
        if cls._default is not NoDefault:
            assert opt is not None
            cls._defaults = copy.deepcopy(cls._defaults)
            cls._defaults[opt] = copy.deepcopy(cls._default)
        SimplePlantParameter._on_registration(cls)

    @classmethod
    def _get_property_dependencies(cls, name, prefix='', **kwargs):
        out = super(
            OptionPlantParameter, cls)._get_property_dependencies(
                name, prefix=prefix, **kwargs)
        option_property = cls._name_option
        if name == option_property:
            if len(out) > 1:
                out[-1].update(out[-2])
                out[-2] = {}
            return out
        return out

    @classmethod
    def specialize(cls, name, exclude_options=None, **kwargs):
        r"""Created a specialized version of this class.

        Args:
            name (str): Name of the root property for the class.
            exclude_options (list, optional): Set of options to eliminate.
            **kwargs: Additional keyword arguments are added as attributes
                to the resulting class.

        """
        if exclude_options:
            kwargs['_option_dependencies'] = copy.deepcopy(
                cls._option_dependencies)
            kwargs.setdefault('exclude', [])
            for x in exclude_options:
                kwargs['exclude'] += list(
                    cls.option_parameters(x, unique=True)
                )
                kwargs['_option_dependencies'].pop(x)
        return super(OptionPlantParameter, cls).specialize(name, **kwargs)

    @classmethod
    def schema(cls):
        r"""Create a JSON schema for parsing parameters used by this
        parameter.

        Returns:
            dict: JSON schema.

        """
        opt = cls._name_option
        out = super(OptionPlantParameter, cls).schema()
        out['oneOf'] = []
        for k, v in cls._option_dependencies.items():
            kprop = {}
            krequired = out.get('required', []) + v
            assert opt in krequired
            for kreq in krequired:
                kprop[kreq] = copy.deepcopy(out['properties'][kreq])
            kprop[opt]['enum'] = [k]
            if cls._default is NoDefault:
                unique_param = cls.option_parameters(k, unique=True)
                if unique_param:
                    kprop[opt]['default'] = k
            out['oneOf'].append({
                'properties': kprop,
                'required': krequired,
            })
        option_properties = cls.option_parameters(None, unique=True)
        for k in option_properties:
            out['properties'].pop(k)
        out.pop('required', None)
        return out

    @classmethod
    def option_parameters(cls, option, unique=False, required=False):
        r"""Get the set of parameters enabled by an option.

        Args:
            option (str): Option value.
            unique (bool, optional): If True, return only those parameters
                that are unique to the provided option.

        Returns:
            set: Option parameters.

        """
        if required:
            all_options = (
                set(cls._required) - set([cls._name_option])
            )
        else:
            all_options = (
                set(cls._properties.keys()) - set([cls._name_option])
            )
        if option is None and not (unique | required):
            return all_options
        other_options = set()
        this_option = set()
        for k, v in cls._option_dependencies.items():
            vall = set(v)
            if k == option:
                this_option |= vall
            else:
                other_options |= vall
        if option is None and unique:
            return other_options
        elif option is None and required:
            return all_options | other_options
        elif unique:
            return this_option - other_options
        return (all_options - other_options) | this_option

    @classmethod
    def add_class_defaults(cls, param, **kwargs):
        r"""Update defaults based on parameters.

        Args:
            param (ParameterDict): Parameter set to update.
            **kwargs: Additional keyword arguments are added to param.

        """
        # Remove defaults added for unused options
        opt = cls._name_option
        vopt = param.get(opt, NoDefault)
        unique_keys = {
            k: cls.option_parameters(k, unique=True)
            for k in cls._option_dependencies.keys()
        }
        if vopt is NoDefault:
            choices = {}
            for k, unique in unique_keys.items():
                count = sum([
                    x in param for x in unique
                ])
                if unique and count:
                    choices[k] = count
            if choices:
                vopt = max(choices, key=lambda x: choices[x])
                kwargs.setdefault(opt, vopt)
        super(OptionPlantParameter, cls).add_class_defaults(
            param, **kwargs)


class FunctionPlantParameter(OptionPlantParameter):
    r"""Class for a function."""

    _name = 'Func'
    _help = 'Function producing {parent}'
    _properties = {
        'Var': {
            'type': 'string',
            'description': ('Parameter that serves as the {parent} '
                            'function\'s independent variable'),
        },
        'VarNorm': {
            'type': 'string',
            'description': ('Parameter that {parent}Var '
                            'should be normalized by before applying '
                            'the {parent} function'),
        },
        'VarMin': {
            'type': 'string',
            'description': ('Parameter that specifies the minimum '
                            '{parent}Var value for which the {parent} '
                            'function is applied.'),
        },
        'VarMax': {
            'type': 'string',
            'description': ('Parameter that specifies the maximum '
                            '{parent}Var value for which the {parent} '
                            'function is applied.'),
        },
        'Slope': {
            'type': 'number',
            'description': 'Slope of the {parent} function',
        },
        'Intercept': {
            'type': 'number',
            'description': ('Value of the {parent} function when '
                            '{parent}Var is 0'),
        },
        'Amplitude': {
            'type': 'number', 'default': 1.0,
            'description': 'Scale factor for {parent} function',
        },
        'Period': {
            'type': 'number', 'default': 2.0 * np.pi,
            'description': 'Period of {parent} sinusoid function',
        },
        'XOffset': {
            'type': 'number', 'default': 0.0,
            'description': ('Offset added to independent variable '
                            'before scaling for the period and applying '
                            'the sinusoid function'),
        },
        'YOffset': {
            'type': 'number', 'default': 0.0,
            'description': ('Offset added to the function result'),
        },
        'Exp': {
            'type': 'number',
            'description': 'Exponent',
        },
        'XVals': {
            'oneOf': [
                {'type': 'ndarray', 'subtype': 'float'},
                {'type': 'array', 'items': {'type': 'number'},
                 'minItems': 2, 'maxItems': 2},
            ],
            'description': ('Independent variable values or range that '
                            'should be used for interpolation'),
        },
        'YVals': {
            'type': 'ndarray', 'subtype': 'float',
            'description': ('Dependent variable values that should be '
                            'used for interpolation'),
        },
        'Curve': DelayedPlantParameter(
            'curve',
            _help='Curve that should be sampled as a function',
        ),
        'CurvePatch': DelayedPlantParameter(
            'curve_patch',
            _help='Curve patch that should be sampled as a function',
        ),
        'Method': {
            'type': 'string',
            'description': ('Name of the method on the plant generator '
                            'class that should be called')
        },
        'Function': {'type': 'function'},
    }
    _option_dependencies = {
        'linear': ['Slope', 'Intercept'],
        'sin': ['Amplitude', 'Period', 'XOffset', 'YOffset'],
        'cos': ['Amplitude', 'Period', 'XOffset', 'YOffset'],
        'tan': ['Amplitude', 'Period', 'XOffset', 'YOffset'],
        'pow': ['Amplitude', 'Exp', 'XOffset', 'YOffset'],
        'interp': ['XVals', 'YVals'],
        'curve': ['Curve'],
        'curve_patch': ['CurvePatch'],
        'method': ['Method'],
        'user': ['Function'],
    }
    _required = ['Var']
    _variables = ['Var']

    @property
    def xvar(self):
        r"""str: Variable that this function parameter takes as input."""
        return self.parameters['Var']

    @property
    def normvar(self):
        r"""str: Variable that should be used to normalize the input."""
        return self.parameters.get('VarNorm', None)

    @property
    def maxvar(self):
        r"""str: Variable that contains the maximum value under which
        the function applies."""
        return self.parameters.get('VarMax', None)

    @property
    def minvar(self):
        r"""str: Variable that contains the minimum value under which
        the function applies."""
        return self.parameters.get('VarMin', None)

    def generate(self, idx, **kwargs):
        r"""Generate this parameter.

        Args:
            idx (tuple): Index variables that should be used to generate
                this parameter.
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        kwargs['idx'] = idx
        func = self.parameters['']
        v = self.root.get(self.xvar, k0=self.xvar, **kwargs)
        if self.maxvar is not None:
            vmax = self.get(self.maxvar, k0=self.maxvar, **kwargs)
            if v > vmax:
                return 1.0
        if self.minvar is not None:
            vmin = self.get(self.minvar, k0=self.minvar, **kwargs)
            if v < vmin:
                return 1.0
        if self.normvar is not None:
            v /= float(self.get(self.normvar, k0=self.normvar, **kwargs))
        if callable(func):
            out = func(v)
        elif func == 'linear':
            slope = self.parameters['Slope']
            intercept = self.parameters['Intercept']
            out = slope * v + intercept
        elif func in ['sin', 'cos', 'tan']:
            A = self.parameters['Amplitude']
            period = self.parameters['Period']
            xoffset = self.parameters['XOffset']
            yoffset = self.parameters['YOffset']
            ftrig = getattr(np, func)
            out = (
                (A * ftrig(2.0 * np.pi * (v + xoffset) / period))
                + yoffset)
        elif func == 'pow':
            A = self.parameters['Amplitude']
            exp = self.parameters['Exp']
            xoffset = self.parameters['XOffset']
            yoffset = self.parameters['YOffset']
            out = (A * pow(v + xoffset, exp)) + yoffset
        elif func == 'interp':
            xvals = self.parameters['XVals']
            yvals = self.parameters['YVals']
            if isinstance(xvals, (tuple, list)):
                xvals = np.linspace(xvals[0], xvals[1], len(yvals))
            f = scipy.interpolate.interp1d(xvals, yvals)
            out = f(v)
        elif func == 'curve':
            curve = self.get('Curve', **kwargs)
            out = CurvePlantParameter.sample_curve(curve, v)
        elif func == 'curve_patch':
            patch = self.get('CurvePatch', **kwargs)
            out = CurvePatchPlantParameter.sample_curve_patch(patch, v)
        elif func == 'method':
            method = self.get('Method', **kwargs)
            assert self.root.hasmethod(method)
            out = getattr(self.root, method)(v)
        elif func == 'user':
            function = self.get('Function', **kwargs)
            out = function(v)
        else:
            raise ValueError(f"Unsupported function name \"{func}\"")
        self.log(f'{self.fullname} = {out}')
        return out

    @classmethod
    def specialize_var(cls, var, normvar=None, minvar=None, maxvar=None,
                       **kwargs):
        r"""Created a specialized version of this class.

        Args:
            var (str): Name of the variable that the specialized
                function is dependent on.
            normvar (str, optional): Name of the variable that should be
                used to normalize the dependent variable.
            maxvar (str, optional): Name of the variable that should be
                used to cap the values of the dependent variable on
                which the function should be evaluated.
            **kwargs: Additional keyword arguments are passed to the
                base class's specialize method.

        """
        kwargs.setdefault('xvar', var)
        kwargs.setdefault('normvar', normvar)
        kwargs.setdefault('minvar', minvar)
        kwargs.setdefault('maxvar', maxvar)
        var = kwargs['xvar']
        descriptions = {
            'norm': f'Value used to normalize {var.lower()} for {{parent}}',
            'min': f'Minimum {var.lower()} over which {{parent}} is valid',
            'max': f'Maximum {var.lower()} over which {{parent}} is valid',
        }
        kwargs.setdefault('required', [])
        kwargs.setdefault('properties', {})
        kwargs.setdefault('exclude', [])
        kwargs['exclude'] += [
            'Var', 'VarNorm', 'VarMin', 'VarMax',
        ]
        for k in descriptions.keys():
            x = kwargs[f'{k}var']
            if x is None:
                continue
            if x not in kwargs['required']:
                kwargs['required'].append(x)
            if x not in kwargs['properties']:
                kwargs['properties'][x] = {
                    'type': 'number',
                    'description': descriptions[k],
                }
        out = FunctionPlantParameter.specialize(f'{var}Func', **kwargs)
        if var not in out._dependencies:
            out._dependencies = copy.deepcopy(out._dependencies)
            out._dependencies.append(var)
        return out


class DistributionPlantParameter(OptionPlantParameter):
    r"""Class for a distribution."""

    _name = 'Dist'
    _help = 'Distribution applied to {parent}'
    _properties = {
        'Mean': DelayedPlantParameter(
            'scalar', specialize_name='Mean',
            exclude=['Dist'],
            _help='Mean of {parents[1]} normal distribution',
        ),
        'StdDev': DelayedPlantParameter(
            'scalar', specialize_name='StdDev',
            exclude=['Dist'],
            _help=('Standard deviation of {parents[1]} normal '
                   'distribution'),
        ),
        'Bounds': {
            'type': 'array', 'items': {'type': 'number'},
            'minItems': 2, 'maxItems': 2,
            'description': ('Minimum and maximum of the {parents[0]} '
                            'distribution'),
        },
    }
    _option_dependencies = {
        'normal': ['Mean', 'StdDev'],
        'uniform': ['Bounds'],
    }

    @property
    def profile(self):
        r"""str: Distribution profile."""
        return self.parameters['']

    def generate(self, idx, **kwargs):
        r"""Generate this parameter.

        Args:
            idx (tuple): Index variables that should be used to generate
                this parameter.
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        profile = self.profile
        kws = {k: self.get(k, **kwargs) for k in
               self._option_dependencies[profile]}
        kwargs = dict(kws, **kwargs)
        out = self.sample_generator_dist(
            self.generator, profile=profile, **kwargs
        )
        self.log(f'{self.fullname} = {out}')
        return out

    @classmethod
    def sample_generator_dist(cls, generator, profile='normal',
                              args=None, **kwargs):
        r"""Sample a distribution using the current random number
        generator.

        Args:
            generator (np.random.Generator): Generator to use.
            profile (str): Name of the profile that should be sampled.
            args (tuple, optional): Positional arguments to pass to the
                generator method. If not provided, the profile parameters
                must be provided by kwargs.
            **kwargs: Additional keyword arguments are checked for the
                required profile parameters if args is not provided. Any
                remaining keyword arguments are passed to the generator
                method.

        """
        if args is None:
            if profile == 'uniform':
                args = kwargs.pop('Bounds')
            else:
                args = [kwargs.pop(k) for k in
                        cls._option_dependencies[profile]]
        return getattr(generator, profile)(*args, **kwargs)

    @classmethod
    def parametrize_dist(cls, values, profile='normal', **kwargs):
        r"""Parameterize a distribution of values.

        Args:
            values (pandas.DataFrame, np.ndarray): Set of values.
            profile (str, optional): Distribution profile that should be
                parameterized.
            **kwargs: Additional keyword arguments are passed to the
                methods used to determine the distribution parameters.

        Returns:
            tuple: Set of parameters for the distribution.

        """
        if isinstance(values, pd.DataFrame):
            values = values.to_numpy()
        if profile in ['normal', 'gauss', 'gaussian']:
            param = {
                'Mean': np.nanmean(values, **kwargs),
                'StdDev': np.nanstd(values, **kwargs),
            }
        # elif profile in ['choice']:
        #     param = (values, )
        elif profile in ['uniform']:
            param = {
                'Bounds': np.nanmin(values, **kwargs),
            }
        else:
            raise ValueError(f"Unsupported profile \"{profile}\"")
        return param


class ScalarPlantParameter(SimplePlantParameter):
    r"""Class for scalar parameters that will have a spread by default
    and can have a dependence on age, n, or x."""

    _name = 'scalar'
    _unit_dimension = None
    _properties = {
        '': {
            'type': 'number',
            'description': 'Mean of {parent}',
        },
        'StdDev': DelayedPlantParameter(
            'scalar', specialize_name='StdDev',
            exclude=['Dist', 'StdDev', 'RelStdDev'],
            _help=('Standard deviation of {parents[1]} normal '
                   'distribution'),
        ),
        'RelStdDev': DelayedPlantParameter(
            'scalar', specialize_name='RelStdDev',
            exclude=['Dist', 'StdDev', 'RelStdDev'],
            _help=('Standard deviation of {parents[1]} normal '
                   'distribution relative to the mean'),
        ),
        'Dist': DistributionPlantParameter.specialize(
            'scalar_dist',
            _help='Distribution that {parents[0]} should be scaled by',
            _dependencies=['X', 'N', 'Age'],  # Force update for each set
            exclude_options=['normal'],
        ),
        'Func': FunctionPlantParameter.specialize(
            'scalar_func',
            _help='Function that {parents[0]} should be scaled by',
        ),
        'XFunc': FunctionPlantParameter.specialize_var(
            'X',
            _help=('Dependence of {parents[0]} on the distance along the '
                   '{component}'),
        ),
        'NFunc': FunctionPlantParameter.specialize_var(
            'N', normvar='NMax',
            component_properties=['NMax'],
            _help=('Dependence of {parents[0]} on the {component} '
                   'phytomer number'),
        ),
        'AgeFunc': FunctionPlantParameter.specialize_var(
            'Age', normvar='AgeMature', maxvar='AgeMature',
            properties={
                'AgeSenesce': {
                    'type': 'number',
                    'description': (
                        'Maximum age over which {parent} is valid'
                    ),
                },
            },
            component_properties=['AgeSenesce', 'AgeMature'],
            # defaults={'Slope': 1.0, 'Intercept': 0.0},
            # _default='linear',
            _help=('Dependence of {parents[0]} on the {component} age'),
        ),
    }
    _required = ['']
    _modifiers = [
        'Dist', 'Func', 'XFunc', 'NFunc', 'AgeFunc'
    ]
    _property_dependencies = dict(
        SimplePlantParameter._property_dependencies,
        **{k: {'': True} for k in [
            'Dist', 'Func', 'XFunc', 'NFunc', 'AgeFunc'
        ]}
    )
    _property_dependencies_defaults = dict(
        SimplePlantParameter._property_dependencies_defaults,
        **{'': {
            'value': 1.0,
            'conditions': {
                k: True for k in [
                    'Dist', 'Func', 'XFunc', 'NFunc', 'AgeFunc'
                ]
            },
        }}
    )

    @property
    def dependencies(self):
        r"""list: Set of variables that this parameter is dependent on."""
        out = super(ScalarPlantParameter, self).dependencies
        if 'StdDev' in self.parameters or 'RelStdDev' in self.parameters:
            # Force update for each set
            out += [k for k in ['X', 'N', 'Age'] if k not in out]
        return out

    def generate(self, idx, **kwargs):
        r"""Generate this parameter.

        Args:
            idx (tuple): Index variables that should be used to generate
                this parameter.
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        base = self.parameters.get('', 1.0)
        self.log(f'base = {base}')
        stddev = self.get('StdDev', None, idx=idx)
        out = base
        for k in self._modifiers:
            v = self.parameters.get(k, None)
            if v and v.initialized:
                ifactor = v.generate(idx, **kwargs)
                self.log(f'{k} = {ifactor}')
                out *= ifactor
        if stddev is None:
            relstddev = self.get('RelStdDev', None, idx=idx)
            if relstddev is not None:
                stddev = np.abs(relstddev * out)
        if stddev is not None:
            self.log(f"SAMPLE STDDEV: {out}, {stddev}")
            try:
                out = self.sample_dist('normal', Mean=out, StdDev=stddev)
            except BaseException:
                self.log(f"SAMPLE STDDEV: {out}, {stddev}", force=True)
                raise
        self.log(f'{self.fullname} = {out}')
        return out


class CurvePlantParameter(SimplePlantParameter):
    r"""Class for a curve parameter.

    ClassAttributes:
        _required_curve (list): Properties required when the curve is
            not generated from a curve patch.
        _required_patch (list): Properties required when the curve is
            generated from a curve patch.
        _allow_patch (bool): True if curve patches should be allowed.

    """

    _name = 'curve'
    _properties = {
        'ControlPoints': {
            'type': 'ndarray', 'subtype': 'float', 'ndim': 2,
            'description': '2D control points for the {parent} curve'
        },
        'Symmetry': {
            'type': 'array', 'items': {'type': 'integer'},
            'description': 'Dimensions of the {parent} curve symmetry',
        },
        'Closed': {
            'type': 'boolean', 'default': False,
            'description': 'Should the the {parent} curve be closed',
        },
        'Reverse': {
            'type': 'boolean', 'default': False,
            'description': ('Should the {parent} control points be '
                            'added to the curve in reverse order'),
        },
        'Thickness': {
            'type': 'number',
            'description': ('Double the {parent} curve back on itself '
                            'with this thickeness'),
        },
        'Patch': DelayedPlantParameter(
            'curve_patch',
            _help=('Curve patch that should be sampled to generate the '
                   '{parent} curve'),
        ),
        'PatchVar': {
            'type': 'string',
            'description': ('Parameter that should be used to sample '
                            'the {parent}Patch'),
        },
        'PatchNorm': {
            'type': 'number',
            'description': ('Value that should be used to normalize '
                            '{parent}PatchVar prior to '
                            'sampling {parent}Patch'),
        },
        'PatchMin': {
            'type': 'number',
            'description': ('Minimum value over which '
                            '{parent}PatchVar is valid'),
        },
        'PatchMax': {
            'type': 'number',
            'description': ('Maximum value over which '
                            '{parent}PatchVar is valid'),
        },
    }
    _property_dependencies = {
        'Symmetry': {'ControlPoints': True},
        'Closed': {'ControlPoints': True},
        'Reverse': {'ControlPoints': True},
        'Thickness': {'ControlPoints': True},
        'PatchVar': {'Patch': True},
        'PatchNorm': {'Patch': True, 'PatchVar': True},
        'PatchMin': {'Patch': True, 'PatchVar': True},
        'PatchMax': {'Patch': True, 'PatchVar': True},
    }
    _required = []
    _required_curve = ['ControlPoints']
    _required_patch = ['Patch', 'PatchVar']
    _allow_patch = True

    @staticmethod
    def _on_registration(cls):
        SimplePlantParameter._on_registration(cls)
        cls._patch_properties = [k for k in cls._properties.keys()
                                 if k.startswith('Patch')]
        cls._curve_properties = cls._required_curve
        cls._allow_patch = ('Patch' in cls._properties)
        if cls._allow_patch:
            cls._shared = [
                k for k in cls._properties.keys()
                if k not in
                cls._patch_properties + cls._curve_properties
            ]
            cls._variables = [
                k for k in cls._variables
                if k not in cls._curve_properties
            ] + ['PatchVar']

    @classmethod
    def schema(cls):
        r"""Create a JSON schema for parsing parameters used by this
        parameter.

        Returns:
            dict: JSON schema.

        """
        out = super(CurvePlantParameter, cls).schema()
        if not cls._allow_patch:
            out['required'] = copy.deepcopy(cls._required_curve)
            return out
        out['oneOf'] = []
        for opt in ['curve', 'patch']:
            optreq = getattr(cls, f'_required_{opt}')
            out['oneOf'].append({
                'type': 'object',
                'properties': {
                    k: copy.deepcopy(out['properties'][k])
                    for k in optreq
                },
                'required': copy.deepcopy(optreq),
            })
            for k in optreq:
                out['properties'].pop(k)
        return out

    @classmethod
    def add_class_defaults(cls, param, **kwargs):
        r"""Update defaults based on parameters.

        Args:
            param (ParameterDict): Parameter set to update.
            **kwargs: Additional keyword arguments are added to param.

        """
        super(CurvePlantParameter, cls).add_class_defaults(
            param, **kwargs)
        # TODO: Verify that curve properties arn't remove due to
        #   property_dependencies in parent method
        if cls._allow_patch and not all(k in param for k in
                                        cls._required_curve):
            for k in cls._shared:
                v = param.get(k, NoDefault)
                if v is NoDefault:
                    continue
                for kchild in ['Start', 'End']:
                    param.setdefault(f'Patch{kchild}Curve{k}', v)

    def generate(self, idx, return_points=False, **kwargs):
        r"""Generate this parameter.

        Args:
            idx (tuple): Index variables that should be used to generate
                this parameter.
            return_points (bool, optional): If True, the control points
                for the curve will be returned instead of the curve.
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        kwargs['idx'] = idx
        if ((self._allow_patch and 'Patch' in self.parameters
             and self.parameters['Patch'].initialized)):
            if return_points:
                kwargs.pop('idx')
                patch = self.parameters['Patch']
                return (
                    patch.parameters('StartCurve').generate(
                        idx, return_points=return_points, **kwargs),
                    patch.parameters('EndCurve').generate(
                        idx, return_points=return_points, **kwargs),
                )
            patch = self.get('Patch', **kwargs)
            tvar = self.get('PatchVar', **kwargs)
            tnorm = self.parameters.get('PatchNorm', None)
            tmin = self.parameters.get('PatchMin', None)
            tmax = self.parameters.get('PatchMax', None)
            t = self.get(tvar, **kwargs)
            curve = CurvePatchPlantParameter.sample_curve_patch(
                patch, t, tnorm=tnorm, tmin=tmin, tmax=tmax)
        else:
            points = self.get('ControlPoints', **kwargs)
            symmetry = self.parameters.get('Symmetry', None)
            closed = self.parameters.get('Closed', False)
            reverse = self.parameters.get('Reverse', False)
            thickness = self.get('Thickness', None)
            curve = self.create_curve(
                points, symmetry=symmetry, closed=closed,
                reverse=reverse, thickness=thickness,
                return_points=return_points,
            )
        self.log(f'{self.fullname} = {curve}')
        return curve

    @classmethod
    def reflect_points(cls, points_R, axis):
        r"""Reflect points across an axis.

        Args:
            points_R (np.ndarray): Points that should be reflected from
                the positive side of the specified axis to the negative
                side.
            axis (int): Index of axis that points should be reflected
                across.

        Returns:
            np.ndarray: Set of points including the original points and
                the reflected points.

        """
        scale = np.ones((points_R.shape[1], ))
        scale[axis] = -1.0
        points_L = points_R[::-1] * scale
        if np.allclose(points_L[-1, :], points_R[0, :]):
            points_L = points_L[:-1, :]
        return np.concatenate([points_L, points_R])

    @classmethod
    def sample_curve(cls, curve, t):
        r"""Sample a curve.

        Args:
            curve (NurbsCurve): Curve to sample.
            t (float): Value to sample the curve at.

        Returns:
            float: Curve value at value t.

        """
        raise NotImplementedError("Sampling curve")

    @classmethod
    def create_curve(cls, points, knots=None, uniform=False,
                     stride=60, degree=3, symmetry=None, closed=False,
                     reverse=False, thickness=None, factor=None,
                     return_points=False):
        r"""Create a PlantGL NURBS curve.

        Args:
            points (np.ndarray): Control points. If points are 2D a
                NurbsCurve2D instance will be created, otherwise a
                NurbsCurve instance will be created.
            knots (list, optional): Knot list.
            uniform (bool, optional): If True, the spacing between control
                points is made to be uniform.
            stride (int, optional): The curve stride.
            degree (int, optional): The curve degree.
            symmetry (list, optional): The indices of axes that the points
                should be reflected over.
            closed (bool, optional): If True, the curve is closed.
            reverse (bool, optional): If True, the order of the points
                should be reversed.
            thickness (float, optional): Thickness that should be added
                to the curve by doubling it back on itself.
            factor (float, optional): Additional scale factor that should
                be applied to points.
            return_points (bool, optional): If True, the control points
                for the curve will be returned instead of the curve.

        Returns:
            NurbsCurve: Curve instance.

        """
        if factor is not None:
            points = factor * points
        if symmetry is not None:
            for ax in symmetry:
                points = cls.reflect_points(points, ax)
        if reverse:
            points = points[::-1, :]
        if thickness is not None:
            closed = True
            if points.shape[1] != 2:
                raise NotImplementedError('Thickness for 3D curve')
            nthickness1 = int(np.ceil(points.shape[0] / 2))
            nthickness2 = int(np.floor(points.shape[0] / 2))
            assert (nthickness1 + nthickness2) == points.shape[0]
            thickness1 = np.linspace(0, thickness, nthickness1)
            thickness2 = thickness1[:nthickness2]
            thickness0 = np.concatenate([thickness1, thickness2[::-1]])
            points_bottom = points.copy()
            points_bottom[:, 1] -= thickness0
            points_bottom = points_bottom[::-1, :]
            points = np.concatenate([
                points[:1, :], points,
                points_bottom, points_bottom[-1:, :],
            ])
        if closed and points.shape[1] == 2:
            idx0 = np.argmax(points[:, 1])
            points = np.concatenate([
                points[idx0:, :], points[:idx0, :]
            ])
        if closed and not np.allclose(points[0, :], points[-1, :]):
            points = np.concatenate([points, points[:1, :]])
        if return_points:
            return points
        use2D = False
        if points.shape[1] == 2:
            use2D = True
            ctrl_points = [Vector2(*vec) for vec in points]
        else:
            ctrl_points = [Vector3(*vec) for vec in points]
        nb_pts = len(ctrl_points)
        nb_arc = (nb_pts - 1) // degree
        # nb_knots = degree + nb_pts
        p = 0.
        param = [p]
        for i in range(nb_arc):
            if uniform:
                p += 1
            else:
                p += pglmath.norm(
                    ctrl_points[degree * i]
                    - ctrl_points[degree * (i + 1)]
                )
            param.append(p)
        if knots is None:
            knots = [param[0]]
            for p in param:
                for j in range(degree):
                    knots.append(p)
            knots.append(param[-1])
        if use2D:
            out = NurbsCurve2D(
                [Vector3(v[0], v[1], 1.) for v in ctrl_points],
                knots, degree, stride,
            )
        else:
            out = NurbsCurve(
                [Vector4(v[0], v[1], v[2], 1.) for v in ctrl_points],
                knots, degree, stride,
            )
        return out


class CurvePatchPlantParameter(SimplePlantParameter):
    r"""Class for a curve patch parameter."""

    _name = 'curve_patch'
    _help = 'Set of curves transitioning from one curve to another'
    _properties = {
        'Start': CurvePlantParameter.specialize(
            'start_curve', exclude=CurvePlantParameter._patch_properties,
            _help='Curve defining the start of the patch',
        ),
        'End': CurvePlantParameter.specialize(
            'end_curve', exclude=CurvePlantParameter._patch_properties,
            _help='Curve defining the end of the patch',
        ),
    }
    _required = ['Start', 'End']

    @classmethod
    def sample_curve_patch(cls, p, t, tnorm=None, tmin=None, tmax=None):
        r"""Sample a curve patch.

        Args:
            p (NurbsPatch): Curve patch to sample.
            t (float): Value to sample the patch at.
            tnorm (float, optional): Value to normalize t against.
            tmin (float, optional): Value before which the first patch
                should be used.
            tmax (float, optional): Value after which the last patch
                should be used.

        Returns:
            NurbsCurve2D: Curve corresponding to value t.

        """
        if tnorm is not None:
            t /= tnorm
            if tmin is not None:
                tmin /= tnorm
            if tmax is not None:
                tmax /= tnorm
        if tmin is not None and t < tmin:
            t = 0.0
        elif tmax is not None and t > tmax:
            t = 1.0
        else:
            if tmin is None:
                tmin = 0.0
            if tmax is None:
                tmax = 1.0
            t = (t - tmin) / (tmax - tmin)
        section = p.getIsoUSectionAt(t)
        pts = [(i.x, i.y, i.w) for i in section.ctrlPointList]
        return NurbsCurve2D(pts, section.knotList, section.degree)

    @classmethod
    def create_curve_patch(cls, curves, knots=None, degree=3):
        r"""Create a PlantGL NURBS curve patch.

        Args:
            curves (list): Set of NurbsCurve instances defining the
                patch.
            knots (list, optional): Knot list.
            degree (int, optional): The curve degree.

        Returns:
            NurbsPatch: Curve instance.

        """
        # nbcurves = len(curves)
        # if knots is None:
        #     knots = [i / float(nbcurves - 1) for i in range(nbcurves)]
        # k = [
        #     knots[0] for i in range(degree - 1)
        # ] + knots + [
        #     knots[-1] for i in range(degree - 1)
        # ]
        pts = [
            [(i.x, i.y, 0, 1) for i in c.ctrlPointList] for c in curves
        ]
        ppts = pgl.Point4Matrix(pts)
        return NurbsPatch(ppts, udegree=degree, vdegree=curves[0].degree)

    def generate(self, idx, **kwargs):
        r"""Generate this parameter.

        Args:
            idx (tuple): Index variables that should be used to generate
                this parameter.
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        kwargs['idx'] = idx
        start = self.get('Start', **kwargs)
        end = self.get('End', **kwargs)
        patch = self.create_curve_patch([start, end])
        self.log(f'{self.fullname} = {patch}')
        return patch


class Template1DPlantParameter(SimplePlantParameter):
    r"""Class for loading and scaling a 1D template from a shape file."""

    _name = 'Template1D'
    _help = '1D {component} template loaded from file'
    _ndim = 1
    _properties = {
        '': {
            'type': 'string',
            # 'help': 'File containing a 1D {component} template',
        },
        'Index': {
            'type': 'integer',
            'help': ('Index of the shape that should be used from the '
                     '{parent} file'),
            'default': 0,
        },
        'XScale': ScalarPlantParameter.specialize(
            'x_scale',
            _help='Scale factor for {parent} along the x direction',
        ),
    }

    def generate(self, idx, **kwargs):
        r"""Generate this parameter.

        Args:
            idx (tuple): Index variables that should be used to generate
                this parameter.
            **kwargs: Additional keyword arguments are passed to get and
                generate calls for nested parameters.

        Returns:
            object: Generated parameter value.

        """
        import shapefile
        shp = shapefile.Reader(self.parameters['']).shapes()[
            self.parameters['Index']]
        pts = np.array(shp.points)
        assert pts.ndim == 2
        assert pts.shape[-1] == self._ndim
        for i, x in enumerate('XYZ'):
            if i >= self._ndim:
                break
            pts[:, i] *= self.get(f'{x}Scale', **kwargs)
        self.log(f'{self.fullname} = {pts}')
        return pts


class Template2DPlantParameter(Template1DPlantParameter):
    r"""Class for loading and scaling a 2D template from a shape file."""

    _name = 'Template2D'
    _help = '2D {component} template loaded from file'
    _ndim = 2
    _properties = dict(
        Template1DPlantParameter._properties,
        YScale=ScalarPlantParameter.specialize(
            'y_scale',
            _help='Scale factor for {parent} along the y direction',
        ),
    )


class Template3DPlantParameter(Template2DPlantParameter):
    r"""Class for loading and scaling a 3D template from a shape file."""

    _name = 'Template3D'
    _help = '3D {component} template loaded from file'
    _ndim = 3
    _properties = dict(
        Template2DPlantParameter._properties,
        ZScale=ScalarPlantParameter.specialize(
            'z_scale',
            _help='Scale factor for {parent} along the y direction',
        ),
    )


class ParameterCollection(SimplePlantParameter):
    r"""Class for collection of parameters.

    ClassAttributes:
        _parameters (dict): Mapping between parameter class names and
            lists of parameter names that should be included in this
            collection using that class.

    """

    _name = 'collection'
    _parameters = {}
    _properties = {}

    @staticmethod
    def _on_registration(cls):
        SimplePlantParameter._on_registration(cls)
        cls._properties = copy.deepcopy(cls._properties)
        for k, v in cls._parameters.items():
            kcls = get_class_registry().get('plant_parameter', k)
            if isinstance(v, dict):
                for kk, vv in v.items():
                    cls._properties[kk] = kcls.specialize(kk, **vv)
            else:
                for kk in v:
                    cls._properties[kk] = kcls


###############################################
# COMPONENTS
###############################################

class PlantComponent(OptionPlantParameter):
    r"""Base class for generating plant components."""

    _registry_key = 'plant_component'
    _name = None
    _name_option = 'Method'
    _help = None
    _properties = dict(
        ParameterCollection._properties,
        Method={
            'type': 'string',
            'description': ('Method that should be used to generate '
                            'each {component}'),
        },
        Length=ScalarPlantParameter.specialize(
            'length',
            _help='{Component} length',
            _unit_dimension='length',
        ),
        Width=ScalarPlantParameter.specialize(
            'width',
            _help='{Component} width',
            _unit_dimension='length',
        ),
        Angle=ScalarPlantParameter.specialize(
            'polar_angle',
            _help='{Component} angle from parent forward axis',
            _defaults={'': 0},
            _unit_dimension='angle',
        ),
        RotationAngle=ScalarPlantParameter.specialize(
            'polar_angle',
            _help='{Component} rotation angle around parent forward axis',
            _defaults={'': 0},
            _unit_dimension='angle',
        ),
    )

    @staticmethod
    def _on_registration(cls):
        if cls._name:
            cls._help = cls._name.lower()
        OptionPlantParameter._on_registration(cls)

    @property
    def component(self):
        r"""str: Component that this property belongs to."""
        return self


class LeafComponent(PlantComponent):
    r"""Leaf component."""

    _name = 'Leaf'
    _properties = dict(
        PlantComponent._properties,
        Thickness=ScalarPlantParameter.specialize(
            'leaf_thickness',
            _help='Thickness of each leaf',
            _unit_dimension='length',
        ),
        Profile=CurvePlantParameter.specialize(
            'leaf_profile',
            _help='Profile of leaf cross section',
            defaults={
                'Closed': False,
                'Symmetry': [0],
                'Reverse': True,
                'ControlPoints': np.array([
                    [+0.0,  0.0],
                    [+0.1,  0.0],
                    [+0.2,  0.0],
                    [+0.5,  0.1],
                    [+1.0,  0.2],
                ])
            },
            component_properties=['Thickness'],
        ),
        Bend=ScalarPlantParameter.specialize(
            'leaf_bend',
            _help='Angle that a leaf bends per distance along its length',
            _unit_dimension='angle/length',
            defaults={
                '': 0,
            }
        ),
        Twist=ScalarPlantParameter.specialize(
            'leaf_twist',
            _help='Angle that a leaf twists per distance along its length',
            _unit_dimension='angle/length',
            defaults={
                '': 0,
            }
        ),
        NDivide={
            'type': 'integer', 'default': 10,
            'description': ('The number of segments that the leaf will '
                            'be dividied into'),
        },
        Unfurled={
            'type': 'boolean', 'default': False,
            'description': ('The leaf should be unfurled, starting as a '
                            'cylinder and ending as the specified '
                            'LeafProfile'),
        },
        UnfurledLength={
            'type': 'number', 'default': 0.5,
            'description': ('The fraction of the leaf that should be '
                            'unfurled'),
        },
        Template2D=Template2DPlantParameter,
        Template3D=Template3DPlantParameter,
    )
    _option_dependencies = dict(
        PlantComponent._option_dependencies,
        sweep=[
            'Profile', 'Thickness', 'NDivide', 'Unfurled',
            'UnfurledLength', 'Bend', 'Twist',
        ],
        template2d=['Template2D', 'Thickness'],
        template3d=['Template3D'],
    )
    _property_dependencies = dict(
        PlantComponent._property_dependencies,
        UnfurledLength={'Unfurled': [True]},
    )


class InternodeComponent(PlantComponent):
    r"""Internode component."""

    _name = 'Internode'
    _option_dependencies = dict(
        PlantComponent._option_dependencies,
        cylinder=[],
    )


class BudComponent(PlantComponent):
    r"""Bud component."""

    _name = 'Bud'


###############################################
# GENERATOR
###############################################

class PlantGenerator(ParameterCollection):
    r"""Base class for generating plants.

    ClassAttributes:
        _plant_name (str): Name of the plant that this class will
            generate.

    """
    _registry_key = 'crop'
    _plant_name = None
    _name = None
    _help = None
    _default = 'maize'
    _default_data = None
    _arguments = []
    _properties = dict(
        ParameterCollection._properties,
        id={
            'type': 'string',
            'description': ('ID string to be associated with this set of '
                            'property values (e.g. genotype or line)'),
        },
        AgeMature={
            'type': 'scalar', 'subtype': 'float', 'units': 'days',
            'description': 'Age after which {class_name} matures',
        },
        AgeSenesce={
            'type': 'scalar', 'subtype': 'float', 'units': 'days',
            'description': 'Age after which {class_name} senesces',
        },
        NMax={
            'type': 'integer',
            'description': 'Maximum photomer number',
        },
    )
    _inherited_properties = ['AgeMature', 'AgeSenesce', 'NMax']
    _components = {}
    _skip_arguments = {}
    _unit_dimensions = ['length', 'mass', 'time']

    @staticmethod
    def _on_registration(cls):
        if cls._plant_name is not None:
            cls._help = (f'Generate a 3D geometry for a '
                         f'{cls._plant_name} canopy')
            cls._name = cls._plant_name
        cls._properties = copy.deepcopy(cls._properties)
        cls._required = copy.deepcopy(cls._required)
        for component, component_kws in cls._components.items():
            component_cls = get_class_registry().get('plant_component',
                                                     component)
            component_kws = copy.deepcopy(component_kws)
            for k in cls._inherited_properties:
                if k not in component_cls._properties:
                    component_kws.setdefault('properties', {})
                    component_kws['properties'].setdefault(
                        k, copy.deepcopy(cls._properties[k]))
                component_kws.setdefault('external_properties', [])
                component_kws['external_properties'].append(k)
            cls._properties[component] = component_cls.specialize(
                f'{component}_{cls._name}', **component_kws
            )
            if component not in cls._required:
                cls._required.append(component)
        ParameterCollection._on_registration(cls)
        cls._arguments = []
        cls._add_parameter_arguments(cls, cls)

    def __init__(self, param=None, seed=0, verbose=False, debug=False,
                 debug_param=None, debug_param_prefix=None,
                 unit_system=None, no_class_defaults=False, **kwargs):
        self.seed = seed
        if debug_param is None:
            debug_param = []
        if debug_param_prefix is None:
            debug_param_prefix = []
        self._debug_param = debug_param
        self._debug_param_prefix = debug_param_prefix
        self._verbose = verbose
        self._debugging = debug
        self._no_class_defaults = no_class_defaults
        self.unit_system = UnitSet.from_kwargs(
            kwargs, suffix='_units', pop=True
        )
        if param is None:
            param = kwargs
        else:
            if kwargs:
                self.log(f'Invalid keyword arguments '
                         f'{pprint.pformat(kwargs)}')
            assert not kwargs
        param = ParameterDict(param, logger=self)
        super(PlantGenerator, self).__init__(
            '', param, None, required=True,
        )

    @classmethod
    def _property2component(cls, kname):
        for k in cls._components.keys():
            if kname.startswith(k):
                return k
        return None

    @classmethod
    def ids_from_file(cls, fname):
        r"""Determine all of the available ids from the provided file.

        Args:
            fname (str): Data file.

        Returns:
            list: Crop IDs.

        """
        raise NotImplementedError

    @classmethod
    def parameters_from_file(cls, args, parameters):
        r"""Calculate parameters based on emperical data.

        Args:
            args (ParsedArguments): Parsed arguments.
            parameters (dict): Parameter dictionary to update.

        Returns:
            dict: Set of parameters calculated from args.

        """
        raise NotImplementedError

    @property
    def log_prefix_instance(self):
        r"""str: Prefix to use for log messages emitted by this instance."""
        return self._plant_name

    @property
    def crop_class(self):
        return self.get("id")

    @classmethod
    def add_arguments(cls, parser, only_crop_parameters=False, **kwargs):
        r"""Add arguments associated with this subparser to a parser.

        Args:
            parser (InstrumentedParser): Parser that the arguments
                should be added to.
            only_crop_parameters (list, optional): Set of crop parameters
                that should be added to the parser.
            **kwargs: Additional keyword arguments are passed to
                SubparserBase.add_arguments_static.

        """
        from canopy_factory.cli import SubparserBase
        if only_crop_parameters:
            kwargs.setdefault('include', [])
            kwargs['include'] += only_crop_parameters
        SubparserBase.add_arguments_static(cls, parser, **kwargs)
        if only_crop_parameters:
            for k in only_crop_parameters:
                vparser = parser.get_subparser('crop', cls._name)
                kaction = vparser.find_argument(k)
                kaction.dependencies = None
        if not kwargs.get('only_subparser', False):
            cls._add_ids_from_file(parser)

    @classmethod
    def _add_ids_from_file(cls, parser):
        if not (cls._name and cls._default_data):
            return
        try:
            ids = cls.ids_from_file(cls._default_data)
        except NotImplementedError:
            return
        vparser = parser.get_subparser('crop', cls._name)
        ids_action = vparser.find_argument('id')
        ids_action.choices = (
            ids + ids_action.choices if ids_action.choices
            else ids
        )
