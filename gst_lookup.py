"""
gst_lookup.py

Backward-compatible shim. GST extraction has moved to identifier_lookup.py,
which generalizes the same engine to also extract CIN (see that module's
docstring for the full design). This file re-exports the GST-specific
pieces under their original names so any existing import of
`from gst_lookup import GSTRecord, get_gst_details` keeps working
unchanged.

New code should generally prefer `identifier_lookup.get_company_identifiers()`
directly, especially if it needs both GST and CIN for the same company --
`get_gst_details()` here only returns GST and, per its own docstring, pays
for a full retrieval pass to do so.
"""

from __future__ import annotations

from identifier_lookup import GSTRecord, get_gst_details

__all__ = ["GSTRecord", "get_gst_details"]
