"""Redaction of credential-bearing CLI arguments before they are persisted.

Benchmark reports and run manifests record the exact invocation that produced
them, which is what makes a run reproducible. Verbatim ``sys.argv`` would also
record ``--api-key <secret>`` in cleartext next to a Settings block that
carefully redacts the very same credential.

Credential flags are derived from the parser itself: any option whose ``dest``
is secret-shaped under the single engine-wide rule (``is_secret_field``) has its
value redacted. A new credential flag is therefore covered the moment it is
added, with no list to keep in sync.

Abbreviated spellings are covered too. argparse accepts any unambiguous prefix
of a long option unless the parser opts out with ``allow_abbrev=False``, so
``--api-k <secret>`` and ``--api-k=<secret>`` reach a run exactly like the full
spelling. Matching option strings literally would have let those two through
untouched, so resolution mirrors argparse: an exact option string wins, and
anything that is only a prefix is redacted whenever it could resolve to a
credential option -- including the ambiguous case argparse itself rejects, since
a value that never reaches the parser must still never reach an artifact.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from atagia.core.effective_settings import (
    REDACTED_EMPTY,
    REDACTED_SET,
    is_secret_field,
)

_LONG_OPTION_PREFIX = "--"


def credential_option_strings(parser: argparse.ArgumentParser) -> frozenset[str]:
    """Return the parser's option strings whose value is a credential.

    ``_actions`` is argparse's only way to enumerate a parser's options; neither
    harness uses subparsers, so this is the complete option set.
    """
    return frozenset(
        option
        for action in parser._actions
        if action.dest is not argparse.SUPPRESS and is_secret_field(action.dest)
        for option in action.option_strings
    )


def _all_option_strings(parser: argparse.ArgumentParser) -> frozenset[str]:
    """Every option string the parser declares, credential or not."""
    return frozenset(
        option for action in parser._actions for option in action.option_strings
    )


def _carries_credential(
    option: str,
    *,
    credential_options: frozenset[str],
    all_options: frozenset[str],
) -> bool:
    """Whether ``option`` resolves to a credential-bearing flag.

    Resolution order mirrors argparse: an exact option string is that option and
    nothing else, so a non-credential flag that merely happens to prefix a
    credential one keeps its value in the record. Only when the spelling is not
    an exact option is prefix expansion considered, and then it fails closed.
    """
    if option in all_options:
        return option in credential_options
    # A bare "--" is argparse's end-of-options marker, not an abbreviation of
    # every long option, so it must not swallow the argument after it.
    if not option.startswith(_LONG_OPTION_PREFIX) or option == _LONG_OPTION_PREFIX:
        return False
    return any(candidate.startswith(option) for candidate in credential_options)


def redact_invocation_args(
    argv: Sequence[str],
    parser: argparse.ArgumentParser,
) -> list[str]:
    """Return ``argv`` with credential values replaced by redaction sentinels.

    Handles both argparse spellings, ``--api-key VALUE`` and ``--api-key=VALUE``,
    and every abbreviation of them the parser would accept. Flag names are
    preserved so the invocation stays readable and the presence of a credential
    stays auditable.
    """
    credential_options = credential_option_strings(parser)
    all_options = _all_option_strings(parser)
    redacted: list[str] = []
    redact_next = False
    for argument in argv:
        if redact_next:
            redacted.append(REDACTED_SET if argument else REDACTED_EMPTY)
            redact_next = False
            continue
        option, separator, value = argument.partition("=")
        carries_credential = _carries_credential(
            option,
            credential_options=credential_options,
            all_options=all_options,
        )
        if separator and carries_credential:
            redacted.append(f"{option}={REDACTED_SET if value else REDACTED_EMPTY}")
            continue
        redacted.append(argument)
        redact_next = not separator and carries_credential
    return redacted
