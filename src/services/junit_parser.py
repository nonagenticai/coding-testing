"""Parse pytest and cargo-nextest JUnit XML into ``UnifiedTestCase`` lists.

Both frameworks write the Jenkins JUnit shape (``<testsuites>`` > ``<testsuite>`` >
``<testcase>``). A case's status is decided by its direct children:

* ``<failure>`` -> failed
* ``<error>``   -> errored
* ``<skipped>`` -> skipped
* none of those -> passed

nextest adds ``<rerunFailure>`` (a retry that also failed - the case still carries
its final ``<failure>``) and ``<flakyFailure>`` (a failed attempt of a test that
eventually PASSED - no ``<failure>`` child, so it counts as passed). Neither of
those is a status on its own, which is exactly what the child rule above gives.

Parsing uses defusedxml: the report is written by code under test in a
tenant workspace, so it is untrusted input (no entity expansion, no DTDs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from xml.etree.ElementTree import Element  # nosec B405 - type only; parsing is defusedxml

from defusedxml import ElementTree as SafeET

from models.unified_test_models import CaseStatus, UnifiedTestCase

#: Cap on a single case message so one huge panic cannot dominate the response.
MAX_MESSAGE_CHARS = 4000


#: Child element -> case status, checked in this order.
_STATUS_CHILDREN: tuple[tuple[str, CaseStatus], ...] = (
    ("failure", "failed"),
    ("error", "errored"),
    ("skipped", "skipped"),
)


class JUnitParseError(ValueError):
    """The report is not parseable JUnit XML."""


@dataclass
class ParsedJUnit:
    cases: list[UnifiedTestCase] = field(default_factory=list)
    duration: float | None = None

    def count(self, status: CaseStatus) -> int:
        return sum(1 for c in self.cases if c.status == status)


def _float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _message(el: Element) -> str | None:
    text = (el.get("message") or "").strip()
    body = (el.text or "").strip()
    if body and body != text:
        text = f"{text}\n{body}" if text else body
    return text[:MAX_MESSAGE_CHARS] or None


def _case(el: Element) -> UnifiedTestCase:
    name = el.get("name") or "<unnamed>"
    classname = el.get("classname")
    full = f"{classname}::{name}" if classname else name
    status: CaseStatus = "passed"
    message = None
    for tag, st in _STATUS_CHILDREN:
        child = el.find(tag)
        if child is not None:
            status = st
            message = _message(child)
            break
    return UnifiedTestCase(
        name=full, status=status, message=message, duration=_float(el.get("time"))
    )


def parse_junit(xml_text: str) -> ParsedJUnit:
    """Parse a JUnit document (either a ``<testsuites>`` or a bare ``<testsuite>`` root)."""
    try:
        root = SafeET.fromstring(xml_text)
    except Exception as e:  # noqa: BLE001 - ParseError / defusedxml forbidden constructs
        raise JUnitParseError(f"unparseable JUnit XML: {e}") from e
    if root.tag not in ("testsuites", "testsuite"):
        raise JUnitParseError(f"not a JUnit report: root element <{root.tag}>")

    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    parsed = ParsedJUnit()
    for suite in suites:
        parsed.cases.extend(_case(tc) for tc in suite.iter("testcase"))

    duration = _float(root.get("time"))
    if duration is None and suites:
        times = [t for s in suites if (t := _float(s.get("time"))) is not None]
        duration = sum(times) if times else None
    parsed.duration = duration
    return parsed
