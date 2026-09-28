"""Fixed conversational acceptance cases for source-reference quote selection.

These cases evaluate whether a source quote supports a supplied candidate when
recent conversation or a prior chunk supplies its referent. They do not test
candidate extraction, and they do not contain official holdout material.
"""

from __future__ import annotations

from atagia.core.source_references import SourceReferenceCatalog


def _span(source: str, phrase: str, occurrence: int = 0) -> list[int]:
    """Find an exact phrase and require production word/punctuation boundaries."""
    start = -1
    for _ in range(occurrence + 1):
        start = source.find(phrase, start + 1)
        if start < 0:
            raise ValueError(f"Missing phrase occurrence: {phrase!r}, {occurrence}")
    end = start + len(phrase)
    catalog = SourceReferenceCatalog(source)
    if start not in {anchor.char_start for anchor in catalog.anchors}:
        raise ValueError(f"Start is not a production anchor boundary: {phrase!r}")
    if end not in {anchor.char_end for anchor in catalog.anchors}:
        raise ValueError(f"End is not a production anchor boundary: {phrase!r}")
    return [start, end]


def _candidate(
    source: str,
    candidate_id: str,
    canonical_text: str,
    support_kind: str,
    *,
    exact: tuple[str, ...] = (),
    required: tuple[str, ...] = (),
    allowed: str | None = None,
    note: str,
) -> dict:
    if support_kind not in {"direct", "contextual", "inferred", "unsupported"}:
        raise ValueError(f"Unknown support kind: {support_kind}")
    if support_kind == "unsupported":
        if exact or required or allowed is not None:
            raise ValueError("Unsupported candidates cannot have quote ranges")
        return {
            "candidate_id": candidate_id,
            "canonical_text": canonical_text,
            "support_kind": support_kind,
            "expected_ranges": [],
            "required_ranges": [],
            "allowed_range": None,
            "note": note,
        }
    if not exact or not required or allowed is None:
        raise ValueError(
            "Supported candidates need exact, required, and allowed ranges"
        )
    expected_ranges = [_span(source, phrase) for phrase in exact]
    required_ranges = [_span(source, phrase) for phrase in required]
    allowed_range = _span(source, allowed)
    if not all(
        allowed_range[0] <= start < end <= allowed_range[1]
        for start, end in expected_ranges + required_ranges
    ):
        raise ValueError("Quote range extends outside its allowed local passage")
    if not all(
        exact_start <= required_start < required_end <= exact_end
        for exact_start, exact_end in expected_ranges
        for required_start, required_end in required_ranges
    ):
        raise ValueError("Every expected quote must contain all mandatory evidence")
    return {
        "candidate_id": candidate_id,
        "canonical_text": canonical_text,
        "support_kind": support_kind,
        "expected_ranges": expected_ranges,
        "required_ranges": required_ranges,
        "allowed_range": allowed_range,
        "note": note,
    }


def _case(
    case_id: str,
    source_text: str,
    candidates: list[dict],
    *,
    recent_messages: list[dict] | None = None,
    prior_chunk_context: str | None = None,
    note: str,
) -> dict:
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"Duplicate candidate ID in {case_id}")
    return {
        "case_id": case_id,
        "source_text": source_text,
        "role": "user",
        "recent_messages": recent_messages or [],
        "prior_chunk_context": prior_chunk_context,
        "candidates": candidates,
        "note": note,
    }


def load_cases() -> list[dict]:
    """Return new, fixed, JSON-serializable acceptance cases without API calls."""
    cases: list[dict] = []

    source = "The east one, after 3 p.m."
    cases.append(
        _case(
            "elliptical_loading_dock",
            source,
            [
                _candidate(
                    source,
                    "cand_001",
                    "Use the east loading dock after 3 p.m.",
                    "contextual",
                    exact=(source,),
                    required=("east one", "after 3 p.m."),
                    allowed=source,
                    note="The preceding question identifies the dock; the final dot is part of p.m. and is therefore mandatory.",
                )
            ],
            recent_messages=[
                {
                    "role": "assistant",
                    "content": "Which loading dock should the delivery driver use, and when?",
                }
            ],
            note="An elliptical answer has no dock noun; the recent assistant question supplies it.",
        )
    )

    source = "Keep that ordering for future reports, please."
    cases.append(
        _case(
            "elliptical_default_confirmation",
            source,
            [
                _candidate(
                    source,
                    "cand_001",
                    "Future reports should sort newest first by default.",
                    "contextual",
                    exact=("Keep that ordering for future reports, please", source),
                    required=("that ordering", "future reports"),
                    allowed=source,
                    note="The previous chunk defines newest-first ordering. The terminal period is optional; the future scope must remain.",
                )
            ],
            prior_chunk_context="For the report list, show the newest entries first.",
            note="A short continuation inherits the specific ordering from the prior chunk of the same message.",
        )
    )

    source = "Then rotate the second key tonight, before the morning run."
    cases.append(
        _case(
            "inferred_key_deadline",
            source,
            [
                _candidate(
                    source,
                    "cand_001",
                    "Rotate the second key tonight before the morning run because it expires tomorrow.",
                    "inferred",
                    exact=(
                        "rotate the second key tonight, before the morning run",
                        "rotate the second key tonight, before the morning run.",
                    ),
                    required=("second key", "tonight", "before the morning run"),
                    allowed=source,
                    note="The recent message supplies tomorrow's expiry; the source supplies the action and earlier deadline. Only the terminal period is optional.",
                )
            ],
            recent_messages=[
                {
                    "role": "user",
                    "content": "The second key expires tomorrow, and the first run is at 07:00.",
                },
                {
                    "role": "assistant",
                    "content": "Do you want me to leave the key rotation until tomorrow?",
                },
            ],
            note="A justified inference combines the source instruction with the explicit expiry in recent context.",
        )
    )

    source = "Maya said ‘deploy now,’ but I said to wait until QA signs off."
    cases.append(
        _case(
            "speaker_attribution_and_negation",
            source,
            [
                _candidate(
                    source,
                    "cand_001",
                    "The user wants the deploy to wait for QA approval.",
                    "direct",
                    exact=(
                        "I said to wait until QA signs off",
                        "I said to wait until QA signs off.",
                    ),
                    required=("I said", "wait until QA signs off"),
                    allowed=source,
                    note="The user attribution and QA condition are essential; the final period is optional.",
                ),
                _candidate(
                    source,
                    "cand_002",
                    "The user said to deploy now.",
                    "unsupported",
                    note="'Deploy now' belongs to Maya and is contradicted by the user's own statement.",
                ),
            ],
            note="Distinguish quoted third-party advice from the user's contrary instruction.",
        )
    )

    source = "I use Zed for notes. Our staging database is citrine-dev."
    cases.append(
        _case(
            "separate_facts_with_absent_third",
            source,
            [
                _candidate(
                    source,
                    "cand_001",
                    "The user uses Zed for notes.",
                    "direct",
                    exact=("I use Zed for notes", "I use Zed for notes."),
                    required=("Zed for notes",),
                    allowed="I use Zed for notes.",
                    note="Quote only the notes-tool sentence; terminal punctuation is optional.",
                ),
                _candidate(
                    source,
                    "cand_002",
                    "The staging database is citrine-dev.",
                    "direct",
                    exact=(
                        "Our staging database is citrine-dev",
                        "Our staging database is citrine-dev.",
                    ),
                    required=("staging database", "citrine-dev"),
                    allowed="Our staging database is citrine-dev.",
                    note="The staging qualifier and hyphenated identifier are mandatory; the terminal period is optional.",
                ),
                _candidate(
                    source,
                    "cand_003",
                    "The production database is citrine-dev.",
                    "unsupported",
                    note="The identifier is only asserted for staging; no production database is named.",
                ),
            ],
            note="Two distinct supported candidates coexist with a plausible but unsupported production claim.",
        )
    )

    source = "El del jueves, pero solo después de las seis."
    cases.append(
        _case(
            "spanish_elliptical_time",
            source,
            [
                _candidate(
                    source,
                    "cand_001",
                    "La revisión será el jueves, después de las seis.",
                    "contextual",
                    exact=("El del jueves, pero solo después de las seis", source),
                    required=("jueves", "solo después de las seis"),
                    allowed=source,
                    note="The question supplies 'revisión'; the Thursday and after-six restriction must both appear. Final period is optional.",
                )
            ],
            recent_messages=[
                {
                    "role": "assistant",
                    "content": "¿Qué turno reservamos para la revisión, el martes o el jueves?",
                }
            ],
            note="Spanish elliptical answer with a mandatory time qualification.",
        )
    )

    source = "El resum final, no l’esborrany."
    cases.append(
        _case(
            "catalan_approved_document",
            source,
            [
                _candidate(
                    source,
                    "cand_001",
                    "Es va aprovar el resum final i no l’esborrany.",
                    "contextual",
                    exact=("El resum final, no l’esborrany", source),
                    required=("resum final", "no l’esborrany"),
                    allowed=source,
                    note="The prior question establishes approval. Both the selected final summary and rejected draft are required; final period is optional.",
                )
            ],
            recent_messages=[
                {
                    "role": "assistant",
                    "content": "Quin arxiu es va aprovar a la reunió?",
                }
            ],
            note="Catalan attribution of a selected document without treating the rejected draft as approved.",
        )
    )

    source = "预算表给陈，报告给林。"
    cases.append(
        _case(
            "chinese_distinct_recipients",
            source,
            [
                _candidate(
                    source,
                    "cand_001",
                    "陈 receives the budget sheet.",
                    "direct",
                    exact=("预算表给陈", "预算表给陈，"),
                    required=("预算表给陈",),
                    allowed="预算表给陈，",
                    note="The comma can be included or omitted; it does not change the recipient.",
                ),
                _candidate(
                    source,
                    "cand_002",
                    "林 receives the report.",
                    "direct",
                    exact=("报告给林", "报告给林。"),
                    required=("报告给林",),
                    allowed="报告给林。",
                    note="The final period can be included or omitted; quote the report recipient, not the neighboring budget clause.",
                ),
            ],
            note="Chinese source with independent recipients and punctuation-level anchors.",
        )
    )

    source = "Set `retry.max=3`; leave `timeout_ms=2500` unchanged."
    cases.append(
        _case(
            "literal_code_boundaries",
            source,
            [
                _candidate(
                    source,
                    "cand_001",
                    "Set retry.max to 3.",
                    "direct",
                    exact=("Set `retry.max=3`", "Set `retry.max=3`;"),
                    required=("`retry.max=3`",),
                    allowed="Set `retry.max=3`;",
                    note="The trailing semicolon is optional; the backticks, dot, equals sign, and 3 inside the setting are mandatory.",
                ),
                _candidate(
                    source,
                    "cand_002",
                    "Keep timeout_ms at 2500.",
                    "direct",
                    exact=(
                        "leave `timeout_ms=2500` unchanged",
                        "leave `timeout_ms=2500` unchanged.",
                    ),
                    required=("`timeout_ms=2500`", "unchanged"),
                    allowed="leave `timeout_ms=2500` unchanged.",
                    note="Terminal period is optional; underscore, equals sign, and all digits are mandatory.",
                ),
            ],
            note="Accept harmless clause punctuation without relaxing code syntax or numeric values.",
        )
    )

    long_sections = [
        "For Monday intake, the support desk logged 18 new requests and assigned seven to the operations queue.",
        "On Tuesday, design tested the revised navigation with three volunteers and kept the shorter label for the account page.",
        "Facilities inspected the backup lighting and ordered a replacement for the lamp near the north stairwell.",
        "The translation review found an ambiguous date in the Spanish notice, so editorial sent a corrected draft to legal.",
        "Finance matched travel receipts against the approved monthly budget and returned two invoices for missing descriptions.",
        "The archive group indexed completed tickets by quarter and verified that the original attachments remained readable.",
        "Wednesday's warehouse inventory showed twelve spare scanners; four need charging before the next dispatch.",
        "The mobile team reproduced a slow login on older devices and recorded the network trace for the identity owner.",
        "Procurement compared three bids for replacement headsets and asked the vendors to clarify warranty terms.",
        "The training coordinator scheduled a workshop for new shift leads and circulated the checklist in advance.",
        "Thursday's quality review sampled the latest batch and found one label placed over a serial number.",
        "The documentation team rewrote the setup steps after a new employee missed an implied permission check.",
        "A customer asked whether the export would include archived events; support is waiting for a confirmed answer.",
        "The infrastructure group checked disk capacity before the quarterly backup and recorded comfortable headroom.",
        "The dispatch supervisor moved the afternoon pickup earlier because the loading bay closes at five.",
        "The product analyst summarized interview themes and separated reported pain points from suggested features.",
        "Security reviewed access grants for the contractor cohort and flagged two accounts for owner confirmation.",
        "The field team checked weather and road access before choosing the next inspection site.",
        "The customer success lead prepared a renewal summary that distinguishes committed features from ideas under discussion.",
        "Legal returned the partner agreement with comments on retention periods and asked for a revised data schedule.",
        "The network engineer replaced a faulty switch port and measured stable throughput during the afternoon check.",
        "The reception team updated visitor instructions so delivery drivers enter through the marked service door.",
        "An analyst compared April and May response times and found the largest delay in manual approval handoffs.",
        "The release coordinator confirmed that the mobile build passed accessibility review before scheduling its rollout.",
        "A maintenance contractor repaired the upstairs vent and left the air-flow measurements in the facilities log.",
        "The research group cataloged interview recordings with consent dates and restricted access to the study team.",
        "The billing specialist checked the refund queue and resolved a duplicate ticket without charging the customer twice.",
        "The operations lead asked each owner to confirm open dependencies before Friday's planning meeting.",
        "Final field note: the inspection moved to Girona on 12 May because rain closed the original site.",
    ]
    source = "\n".join(long_sections)
    cases.append(
        _case(
            "long_field_note_near_end",
            source,
            [
                _candidate(
                    source,
                    "cand_001",
                    "The field inspection moved to Girona on 12 May because rain closed the original site.",
                    "direct",
                    exact=(
                        "the inspection moved to Girona on 12 May because rain closed the original site",
                        long_sections[-1],
                    ),
                    required=(
                        "moved to Girona on 12 May",
                        "because rain closed the original site",
                    ),
                    allowed=long_sections[-1],
                    note="The local final sentence contains location, date, and cause. Only its final period is optional; citing the whole long message is too broad.",
                )
            ],
            recent_messages=[
                {
                    "role": "assistant",
                    "content": "Please send the week's operations recap, including the final field inspection.",
                }
            ],
            note="A varied, realistic long recap places the relevant evidence at the end without repeated filler.",
        )
    )

    if len(cases) != 10:
        raise ValueError("Acceptance corpus must remain exactly ten cases")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate acceptance case ID")
    if len(SourceReferenceCatalog(source).anchors) < 500:
        raise ValueError("Long acceptance source is too short")
    return cases
