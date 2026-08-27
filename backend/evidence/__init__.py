"""Authenticated delivery of sensitive passenger evidence images.

Nothing is re-exported here on purpose: binding `router` at package level would
shadow the `evidence.router` submodule and break attribute access to it.
Import from `evidence.router` directly.
"""
