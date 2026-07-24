"""Regression tests for ``turbovec.__version__`` (issue #153)."""
from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version

import turbovec


def test_version_attribute_is_pep440_ish_string():
    assert isinstance(turbovec.__version__, str)
    # PEP 440 release segment: N(.N)*, optionally followed by pre/dev
    # suffixes ("0.8.0", "0.9.0rc1", "0.0.0.dev0" all match).
    assert re.match(r"^\d+(\.\d+)*", turbovec.__version__)


def test_version_is_in_all():
    assert "__version__" in turbovec.__all__


def test_version_matches_dist_metadata():
    try:
        expected = version("turbovec")
    except PackageNotFoundError:
        # Source tree without installed dist metadata: the documented
        # fallback marker.
        expected = "0.0.0.dev0"
    assert turbovec.__version__ == expected
