# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pytest
from packaging.version import Version

from . import compat_decorator
from .compat_decorator import _within_ceiling, recent_major_versions


@pytest.fixture
def ceiling(monkeypatch):
    """Pin the version under test, which is otherwise read from the installed build."""

    def _set(version):
        monkeypatch.setattr(
            compat_decorator,
            "version_under_test",
            lambda: Version(version) if version is not None else None,
        )

    return _set


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("8.0.1", True),
        ("9.0.1", True),
        ("10.0.0", True),
        # Above the build: a release branch owes nothing to versions cut after it.
        ("10.0.1", False),
        ("11.0.0", False),
        ("12.0.0b6", False),
    ],
)
def test_within_ceiling(ceiling, version: str, expected: bool):
    ceiling("10.0.0")
    assert _within_ceiling(Version(version)) is expected


def test_within_ceiling_admits_everything_when_the_build_is_unknown(ceiling):
    # Better to test too much than to silently empty the matrix.
    ceiling(None)
    assert _within_ceiling(Version("99.0.0")) is True


def test_ceiling_follows_the_build_rather_than_a_constant(ceiling, monkeypatch):
    """The cap is read from the version under test, so this file needs no edit to be
    cherry-picked onto a later release branch."""
    monkeypatch.setattr(
        compat_decorator,
        "pylance_stable_versions",
        lambda: [Version(v) for v in ["9.0.1", "10.0.0", "11.0.0", "12.0.0", "13.0.0"]],
    )

    ceiling("10.0.0")
    assert recent_major_versions(2) == ["10.0.0", "9.0.1"]

    ceiling("12.0.0")
    assert recent_major_versions(2) == ["12.0.0", "11.0.0"]


def test_recent_major_versions_skips_rather_than_shortens(ceiling, monkeypatch):
    """Versions above the build are skipped, not counted, so capping costs no
    breadth."""
    ceiling("10.0.0")
    monkeypatch.setattr(
        compat_decorator,
        "pylance_stable_versions",
        lambda: [Version(v) for v in ["8.0.1", "9.0.1", "10.0.0", "11.0.0", "12.0.0"]],
    )

    assert recent_major_versions(3) == ["10.0.0", "9.0.1", "8.0.1"]
