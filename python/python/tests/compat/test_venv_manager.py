# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pytest

from .venv_manager import _lance_namespace_dependency


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.0.1", "lance-namespace<0.7"),
        ("4.0.0b1", "lance-namespace<0.7"),
        ("6.0.0b5", "lance-namespace>=0.7.2,<0.8"),
        ("6.0.0", "lance-namespace>=0.7.2,<0.8"),
        ("7.2.0b5", "lance-namespace>=0.8.0,<0.9"),
        ("7.2.0", "lance-namespace>=0.8.0,<0.9"),
        # Self-describing from 8.0.0 on: pinning here would contradict the wheel's own
        # range and make the install unresolvable.
        ("8.0.0b1", None),
        ("8.0.1", None),
        ("12.0.0b11", None),
    ],
)
def test_lance_namespace_dependency(version: str, expected):
    assert _lance_namespace_dependency(version) == expected
