"""Tests for services/types.py — ErrorCategory enum."""

from services.types import ErrorCategory


class TestErrorCategory:
    """Test ErrorCategory enum values and behavior."""

    def test_all_values_exist(self):
        """All expected error categories exist."""
        expected = [
            "syntax",
            "runtime",
            "timeout",
            "memory",
            "network",
            "permission",
            "configuration",
            "dependency",
            "assertion",
            "unknown",
        ]
        for val in expected:
            assert ErrorCategory(val).value == val

    def test_syntax(self):
        assert ErrorCategory.SYNTAX == "syntax"

    def test_runtime(self):
        assert ErrorCategory.RUNTIME == "runtime"

    def test_timeout(self):
        assert ErrorCategory.TIMEOUT == "timeout"

    def test_memory(self):
        assert ErrorCategory.MEMORY == "memory"

    def test_network(self):
        assert ErrorCategory.NETWORK == "network"

    def test_permission(self):
        assert ErrorCategory.PERMISSION == "permission"

    def test_configuration(self):
        assert ErrorCategory.CONFIGURATION == "configuration"

    def test_dependency(self):
        assert ErrorCategory.DEPENDENCY == "dependency"

    def test_assertion(self):
        assert ErrorCategory.ASSERTION == "assertion"

    def test_unknown(self):
        assert ErrorCategory.UNKNOWN == "unknown"

    def test_is_str_enum(self):
        """ErrorCategory values are strings."""
        for member in ErrorCategory:
            assert isinstance(member, str)
            assert isinstance(member.value, str)

    def test_member_count(self):
        """There are exactly 10 error categories."""
        assert len(ErrorCategory) == 10

    def test_invalid_raises(self):
        """Invalid value raises ValueError."""
        import pytest

        with pytest.raises(ValueError):
            ErrorCategory("nonexistent")
