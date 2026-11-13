"""Strict keyword options shared by the pixelcanvas modules.

``reject_unknown`` turns a silently ignored misspelt option into a TypeError that names the closest valid option;
``unknown_name`` / ``check_choice`` do the same for registry names and enumerated values; ``module_getattr`` gives a
module a did-you-mean AttributeError (one line in the module: ``__getattr__ = module_getattr(__name__, HINTS)``).
"""
import difflib
import inspect
import sys

# spellings people naturally use; each group lists interchangeable names (the first allowed one wins)
_WORDS = (("colour", "color"), ("centre", "center"), ("grey", "gray"))
GROUPS = (("tag", "tags"), ("width", "w"), ("height", "h"), ("radius", "r"), ("opacity", "alpha"),
          ("bg", "background"), ("fg", "foreground", "ink"), ("line_width", "stroke_width", "lw"),
          ("pos", "position", "at", "xy"))
ONE_WAY = {"size": "font_size"}          # given name -> option it means (not the reverse: size is often a real option)
ALIASES = dict(ONE_WAY)
_MAX_LISTED = 15


def _spell(name):
    for a, b in _WORDS:
        if a in name:
            name = name.replace(a, b)
    return name


def _variants(name):
    """The given name, its colour/centre/grey spelling and its singular / plural forms."""
    out = [name]
    for n in (name, _spell(name)):
        for v in (n, n[:-1] if n.endswith("s") else n + "s"):
            if v not in out:
                out.append(v)
    return out


def _synonyms(name):
    out = []
    for g in GROUPS:
        if name in g:
            out += [m for m in g if m != name]
    if name in ONE_WAY:
        out.append(ONE_WAY[name])
    return out


def _resolve(name, allowed, aliases):
    if name in allowed:
        return name
    if aliases and aliases.get(name) in allowed:
        return aliases[name]
    for v in _variants(name):
        if v in allowed:
            return v
        if aliases and aliases.get(v) in allowed:
            return aliases[v]
        for s in _synonyms(v):
            for t in _variants(s):
                if t in allowed:
                    return t
    # compound names: fill_colour -> fill_color, stroke_lw -> stroke_stroke_width ...
    for g in GROUPS:
        for m in g:
            if name.endswith("_" + m):
                for t in g:
                    for u in _variants(name[: -len(m)] + t):
                        if u in allowed:
                            return u
    return None


def normalise(given, aliases=None, allowed=None):
    """Return ``given`` with aliased option names replaced by canonical ones.

    Without ``allowed`` only names in the alias tables are rewritten (colour->color, grey->gray, ...).
    Raises TypeError if a name and its alias are both given."""
    out = {}
    for k, v in given.items():
        c = k
        if allowed is not None:
            c = _resolve(k, allowed, aliases) or k
        else:
            if aliases and k in aliases:
                c = aliases[k]
            else:
                c = _spell(k)
        if c in out:
            raise TypeError("option '%s' given twice (also spelt '%s')" % (c, k))
        out[c] = v
    return out


def _options_text(allowed):
    names = list(allowed)
    if not names:
        return "(none)"
    shown = names[:_MAX_LISTED]
    return ", ".join(shown) + (", …" if len(names) > _MAX_LISTED else "")


def reject_unknown(owner, given, allowed, aliases=None):
    """Normalise aliases in ``given`` and raise TypeError for any name not in ``allowed``.

    Message: "<owner>: unknown option 'tags' — did you mean 'tag'? Options: a, b, c"."""
    allowed = list(allowed)
    aset = set(allowed)
    out = {}
    src = {}
    for k, v in given.items():
        c = _resolve(k, aset, aliases)
        if c is None:
            close = difflib.get_close_matches(k, allowed, n=3, cutoff=0.6)
            hint = (" — did you mean %s?" % " or ".join("'%s'" % m for m in close)) if close else ""
            raise TypeError("%s: unknown option '%s'%s Options: %s" % (
                owner, k, hint if hint else " —", _options_text(allowed)))
        if c in out:
            raise TypeError("%s: option '%s' given twice (also spelt '%s')" % (owner, c, src[c]))
        out[c] = v
        src[c] = k
    return out


def unknown_name(kind, name, valid, hint=None):
    """Return a ValueError for an unknown registry name, with the closest valid names."""
    valid = [str(v) for v in valid]
    close = difflib.get_close_matches(str(name), valid, n=3, cutoff=0.5)
    msg = "unknown %s '%s'" % (kind, name)
    if close:
        msg += " — did you mean %s?" % " or ".join("'%s'" % m for m in close)
    if hint:
        msg += " " + hint
    else:
        msg += " Valid: " + _options_text(sorted(valid))
    return ValueError(msg)


def check_choice(owner, option, value, choices):
    """Return ``value`` if it is one of ``choices``, else raise ValueError naming the closest choice."""
    if value in choices:
        return value
    ch = [str(c) for c in choices]
    close = difflib.get_close_matches(str(value), ch, n=3, cutoff=0.5)
    msg = "%s: invalid %s %r" % (owner, option, value)
    if close:
        msg += " — did you mean %s?" % " or ".join("'%s'" % m for m in close)
    raise ValueError(msg + " Choices: " + _options_text(ch))


def signature_options(func):
    """Keyword names ``func`` accepts (excluding self/cls, *args and **kwargs)."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return []
    return [n for n, p in sig.parameters.items()
            if n not in ("self", "cls") and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]


def strict(func, owner=None, aliases=None):
    """Wrap ``func`` (explicit signature, no ``**kwargs``) so unknown keywords raise the helpful TypeError and
    aliased spellings (colour, tags, w ...) are accepted. Returned unchanged if it takes ``**kwargs``."""
    import functools
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return func
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return func
    names = signature_options(func)
    known = frozenset(names)
    label = owner or getattr(func, "__qualname__", getattr(func, "__name__", "call"))

    @functools.wraps(func)
    def wrapper(*a, **k):
        if k and not known.issuperset(k):
            k = reject_unknown(label, k, names, aliases)
        return func(*a, **k)
    return wrapper


def strict_methods(cls, skip=(), aliases=None, init=True):
    """Apply ``strict`` to the public methods (and ``__init__``) of ``cls`` that have explicit signatures."""
    import types
    for name, fn in list(vars(cls).items()):
        if not isinstance(fn, types.FunctionType) or name in skip:
            continue
        if name.startswith("_") and not (init and name == "__init__"):
            continue
        owner = cls.__name__ if name == "__init__" else cls.__name__ + "." + name
        setattr(cls, name, strict(fn, owner, aliases))


def _norm_name(s):
    return s.lower().replace("_", "")


def module_getattr(module_name, hints=None):
    """Return a module-level ``__getattr__`` that explains a missing name instead of a bare AttributeError.

    Use it as the last line of a kit: ``__getattr__ = module_getattr(__name__, {"rectangle": "use sdf.sd_box(...)"})``.
    The error names the closest public functions of the module (``__all__`` when defined, else every public
    function / class); ``hints`` maps a likely wrong name (lower case) to the advice shown for it. Dunder and
    underscore names (probed by hasattr, copy, pickle, doctest, pytest) raise the plain AttributeError at once."""
    short = module_name.rsplit(".", 1)[-1]
    table = {_norm_name(k): v for k, v in (hints or {}).items()}

    def __getattr__(name):
        if name.startswith("_"):
            raise AttributeError("module %r has no attribute %r" % (module_name, name))
        mod = sys.modules.get(module_name)
        names = list(getattr(mod, "__all__", None) or [])
        if not names and mod is not None:
            import types
            names = [n for n, v in vars(mod).items() if not n.startswith("_") and
                     isinstance(v, (types.FunctionType, type)) and getattr(v, "__module__", None) == module_name]
        names = [n for n in names if n != name]
        g = _norm_name(name)
        close = [n for n in names if _norm_name(n) == g]
        if len(g) >= 3:
            close += [n for n in names if n not in close and (g in _norm_name(n) or
                                                              (len(_norm_name(n)) >= 4 and _norm_name(n) in g))]
        close += [n for n in difflib.get_close_matches(name, names, n=4, cutoff=0.6) if n not in close]
        close = close[:4]
        msg = "%s has no %r" % (short, name)
        hint = table.get(g)
        if hint:
            msg += " — " + hint
            if close:
                msg += " (closest names: %s)" % ", ".join(close)
        elif close:
            msg += " — did you mean %s?" % " or ".join("%s.%s" % (short, n) for n in close)
        else:
            shown = ", ".join(names[:_MAX_LISTED])
            msg += " — public names: %s%s" % (shown, ", ..." if len(names) > _MAX_LISTED else "")
        raise AttributeError(msg)

    return __getattr__
