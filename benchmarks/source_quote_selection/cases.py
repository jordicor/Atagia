"""Independent, source-bound quote selection cases for private development evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from atagia.core.source_references import SourceReferenceCatalog, source_sha256


_EXTRACTION_CASES = (
    Path(__file__).resolve().parents[1] / "memory_extraction_cards" / "cases.jsonl"
)


def _interval(source: str, phrase: str, occurrence: int = 0) -> list[int]:
    """Resolve an authored exact substring to production anchor boundaries."""
    start = -1
    for _ in range(occurrence + 1):
        start = source.find(phrase, start + 1)
        if start < 0:
            raise ValueError(f"Missing occurrence {occurrence} of {phrase!r}")
    end = start + len(phrase)
    catalog = SourceReferenceCatalog(source)
    starts = {anchor.char_start for anchor in catalog.anchors}
    ends = {anchor.char_end for anchor in catalog.anchors}
    if start not in starts or end not in ends:
        raise ValueError(f"Substring is not anchor-aligned: {phrase!r}")
    return [start, end]


def _candidate(
    source: str,
    canonical_text: str,
    *,
    expected: list[tuple[str, int]] | None = None,
    required: list[tuple[str, int]] | None = None,
    allowed: tuple[str, int] | None = None,
    notes: str,
) -> dict:
    if expected is None:
        if required or allowed:
            raise ValueError(
                "Unsupported candidates cannot have required or allowed spans"
            )
        return {
            "candidate_id": "",
            "canonical_text": canonical_text,
            "expected_ranges": [],
            "required_ranges": [],
            "allowed_range": None,
            "notes": notes,
        }
    if not expected or not required or allowed is None:
        raise ValueError(
            "Supported candidates need expected, required, and allowed spans"
        )
    exact = [_interval(source, phrase, occurrence) for phrase, occurrence in expected]
    mandatory = [
        _interval(source, phrase, occurrence) for phrase, occurrence in required
    ]
    limit = _interval(source, *allowed)
    if not all(limit[0] <= start and end <= limit[1] for start, end in mandatory):
        raise ValueError("Required evidence must fit inside the allowed range")
    if not all(limit[0] <= start and end <= limit[1] for start, end in exact):
        raise ValueError("Expected intervals must fit inside the allowed range")
    return {
        "candidate_id": "",
        "canonical_text": canonical_text,
        "expected_ranges": exact,
        "required_ranges": mandatory,
        "allowed_range": limit,
        "notes": notes,
    }


def _case(
    case_id: str,
    category: str,
    source: str,
    candidates: list[dict],
    *,
    kind: str = "authored",
    path: str | None = None,
    origin_id: str | None = None,
    origin_notes: str = "Authored independently for this fixed evaluation corpus.",
) -> dict:
    for number, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = f"cand_{number:03d}"
    return {
        "case_id": case_id,
        "category": category,
        "origin": {
            "kind": kind,
            "path": path,
            "id": origin_id,
            "source_sha256": source_sha256(source),
            "notes": origin_notes,
        },
        "source_text": source,
        "candidates": candidates,
    }


def _fixture_sources() -> dict[str, str]:
    rows = (
        json.loads(line)
        for line in _EXTRACTION_CASES.read_text(encoding="utf-8").splitlines()
    )
    return {row["case_id"]: row["message"] for row in rows}


def _long_document(section_count: int, insert_after: int, evidence: str) -> str:
    """Build a varied operational note; section statements carry real distinct facts."""
    topics = (
        "Intake recorded 18 new requests and assigned seven to the operations team.",
        "The access review found two dormant accounts and scheduled their removal.",
        "Design approved the revised navigation map after three user interviews.",
        "The shipping team received replacement labels before its afternoon cutoff.",
        "Finance reconciled the travel ledger with the approved monthly budget.",
        "The support desk escalated a login issue to the identity service owner.",
        "The training group published a short guide for new shift coordinators.",
        "Quality checked the latest batch against the signed inspection sheet.",
        "The archive team moved completed tickets into the quarterly index.",
        "Facilities tested the backup lights and logged one faulty fixture.",
        "Procurement compared three quotes for the next equipment order.",
        "The translation team reviewed the Spanish notice for ambiguous dates.",
    )
    lines = ["Operations report: each numbered section is a separate status update."]
    for index in range(section_count):
        lines.append(f"Section {index + 1}: {topics[index % len(topics)]}")
        if index == insert_after:
            lines.append(evidence)
    return "\n".join(lines)


def load_cases() -> list[dict]:
    """Return frozen, JSON-serializable cases; never load official holdout data."""
    fixture = _fixture_sources()
    cases: list[dict] = []

    def add_fixture(case_id: str, category: str, candidate_specs: list[dict]) -> None:
        source = fixture[case_id]
        cases.append(
            _case(
                f"fixture_{case_id}",
                category,
                source,
                [_candidate(source, **spec) for spec in candidate_specs],
                kind="faithfully_reused",
                path="benchmarks/memory_extraction_cards/cases.jsonl",
                origin_id=case_id,
                origin_notes="Unchanged message field from an existing synthetic extraction-card fixture; checksum covers that raw field.",
            )
        )

    add_fixture(
        "stable_reply_preference",
        "qualification",
        [
            {
                "canonical_text": "The user prefers concise debugging steps first and details only on request.",
                "expected": [
                    ("I prefer concise steps first, then details only if I ask", 0)
                ],
                "required": [("concise steps first", 0), ("details only if I ask", 0)],
                "allowed": (
                    "For future debugging help, I prefer concise steps first, then details only if I ask.",
                    0,
                ),
                "notes": "Both the ordering and the on-request limit are needed; opening context and final punctuation are optional.",
            }
        ],
    )
    add_fixture(
        "multiple_atomic_facts",
        "multiple_candidates",
        [
            {
                "canonical_text": "The user's current city is Valencia.",
                "expected": [("My current city is Valencia", 0)],
                "required": [("current city", 0), ("Valencia", 0)],
                "allowed": ("My current city is Valencia", 0),
                "notes": "Do not cite the dentist or locker clause for the city fact.",
            },
            {
                "canonical_text": "The user's dentist is Dr. Ramos.",
                "expected": [("my dentist is Dr. Ramos", 0)],
                "required": [("dentist", 0), ("Dr. Ramos", 0)],
                "allowed": ("my dentist is Dr. Ramos", 0),
                "notes": "The dentist clause is independently located after the city clause.",
            },
        ],
    )
    add_fixture(
        "spanish_food_preference",
        "spanish",
        [
            {
                "canonical_text": "Prefiere restaurantes tranquilos y sin música alta cuando viaja.",
                "expected": [
                    (
                        "Prefiero restaurantes tranquilos y sin musica alta cuando viajo",
                        0,
                    )
                ],
                "required": [("restaurantes tranquilos", 0), ("sin musica alta", 0)],
                "allowed": (fixture["spanish_food_preference"], 0),
                "notes": "Both quiet setting and low music are needed; original unaccented source is preserved.",
            }
        ],
    )
    add_fixture(
        "avoid_tables_preference",
        "negation",
        [
            {
                "canonical_text": "The user does not want tables unless requested.",
                "expected": [("don't use tables unless I ask for one", 0)],
                "required": [("don't use tables", 0), ("unless I ask", 0)],
                "allowed": (fixture["avoid_tables_preference"], 0),
                "notes": "The unless clause must remain with the negative preference.",
            }
        ],
    )
    add_fixture(
        "api_language_contract",
        "mixed_language",
        [
            {
                "canonical_text": "Keep API names in English and explain the rest in Spanish.",
                "expected": [
                    (
                        "For API names, keep the original English, but explain the rest in Spanish",
                        0,
                    )
                ],
                "required": [
                    ("API names", 0),
                    ("original English", 0),
                    ("rest in Spanish", 0),
                ],
                "allowed": (fixture["api_language_contract"], 0),
                "notes": "The English exception and Spanish default belong together.",
            }
        ],
    )
    add_fixture(
        "catalan_personal_reply_preference",
        "catalan",
        [
            {
                "canonical_text": "Vol que li responguin en català quan parla de coses personals.",
                "expected": [
                    (
                        "M'agrada que em responguis en català quan parlem de coses personals",
                        0,
                    )
                ],
                "required": [("en català", 0), ("coses personals", 0)],
                "allowed": (fixture["catalan_personal_reply_preference"], 0),
                "notes": "The personal-topic qualification is mandatory.",
            }
        ],
    )
    add_fixture(
        "code_phrase_verbatim",
        "code_punctuation",
        [
            {
                "canonical_text": "The emergency rollback phrase is RIVER-19-BLUE.",
                "expected": [("emergency rollback phrase is RIVER-19-BLUE", 0)],
                "required": [("emergency rollback phrase", 0), ("RIVER-19-BLUE", 0)],
                "allowed": (fixture["code_phrase_verbatim"], 0),
                "notes": "The complete hyphenated value and its role are mandatory.",
            }
        ],
    )
    add_fixture(
        "quoted_third_party_not_user_preference",
        "contradiction",
        [
            {
                "canonical_text": "The user likes cilantro.",
                "expected": [("I actually like cilantro", 0)],
                "required": [("I actually like cilantro", 0)],
                "allowed": ("I actually like cilantro", 0),
                "notes": "Quote the user correction, not the pasted review.",
            },
            {
                "canonical_text": "The user hates cilantro.",
                "expected": None,
                "notes": "The review is third-party text explicitly rejected as the user's preference.",
            },
        ],
    )
    add_fixture(
        "production_database_name",
        "exact_identifier",
        [
            {
                "canonical_text": "The production database is named aurora-main.",
                "expected": [("production database is called aurora-main", 0)],
                "required": [("production database", 0), ("aurora-main", 0)],
                "allowed": (fixture["production_database_name"], 0),
                "notes": "The exact database name requires the production qualifier.",
            }
        ],
    )
    add_fixture(
        "french_cafe_preference",
        "french",
        [
            {
                "canonical_text": "The user prefers quiet cafes with plenty of natural light.",
                "expected": [
                    (
                        "Je préfère les cafés calmes avec beaucoup de lumière naturelle",
                        0,
                    )
                ],
                "required": [("cafés calmes", 0), ("beaucoup de lumière naturelle", 0)],
                "allowed": (fixture["french_cafe_preference"], 0),
                "notes": "Both ambience and lighting need support.",
            }
        ],
    )

    def add_authored(
        case_id: str, category: str, source: str, specs: list[dict]
    ) -> None:
        cases.append(
            _case(
                case_id,
                category,
                source,
                [_candidate(source, **spec) for spec in specs],
            )
        )

    add_authored(
        "absent_support",
        "unsupported",
        "The meeting starts at 09:00. The room is Cedar 2.",
        [
            {
                "canonical_text": "The meeting is on Thursday.",
                "expected": None,
                "notes": "The source gives time and room, but no day.",
            }
        ],
    )
    add_authored(
        "explicit_contradiction",
        "contradiction",
        "The replacement unit is not waterproof; its case is rated for dust only.",
        [
            {
                "canonical_text": "The replacement unit is waterproof.",
                "expected": None,
                "notes": "The source directly denies the candidate.",
            }
        ],
    )
    source = "Archive note: Use the amber queue.\nCurrent setting: Use the amber queue for urgent tickets only."
    add_authored(
        "repeated_identical_phrase",
        "repeated_context",
        source,
        [
            {
                "canonical_text": "The current setting uses the amber queue for urgent tickets only.",
                "expected": [
                    ("Current setting: Use the amber queue for urgent tickets only", 0)
                ],
                "required": [
                    ("Current setting", 0),
                    ("Use the amber queue", 1),
                    ("urgent tickets only", 0),
                ],
                "allowed": (
                    "Current setting: Use the amber queue for urgent tickets only.",
                    0,
                ),
                "notes": "The identical archived phrase is insufficient; cite the second occurrence with its current and urgent-only context.",
            }
        ],
    )
    source = "Facilities assigned the north room to Mara.\nBudget approved a new monitor for Ivo."
    add_authored(
        "separate_candidates",
        "multiple_candidates",
        source,
        [
            {
                "canonical_text": "Mara was assigned the north room.",
                "expected": [("Facilities assigned the north room to Mara", 0)],
                "required": [("north room", 0), ("Mara", 0)],
                "allowed": ("Facilities assigned the north room to Mara.", 0),
                "notes": "The facilities line alone supports this assignment.",
            },
            {
                "canonical_text": "Ivo received approval for a new monitor.",
                "expected": [("Budget approved a new monitor for Ivo", 0)],
                "required": [("new monitor", 0), ("Ivo", 0)],
                "allowed": ("Budget approved a new monitor for Ivo.", 0),
                "notes": "The second line independently supports this approval.",
            },
        ],
    )
    source = "I used to take the early train. Since June I take the late train, except on Fridays."
    add_authored(
        "changed_commute_exception",
        "qualification",
        source,
        [
            {
                "canonical_text": "Since June, the user takes the late train except on Fridays.",
                "expected": [
                    ("Since June I take the late train, except on Fridays", 0)
                ],
                "required": [
                    ("Since June", 0),
                    ("late train", 0),
                    ("except on Fridays", 0),
                ],
                "allowed": ("Since June I take the late train, except on Fridays.", 0),
                "notes": "The older early-train habit is not current; the Friday exception is essential.",
            }
        ],
    )
    source = "No usis la versió de prova. La versió estable només funciona amb el servidor nou."
    add_authored(
        "catalan_two_clauses",
        "catalan",
        source,
        [
            {
                "canonical_text": "La versió estable només funciona amb el servidor nou.",
                "expected": [
                    ("La versió estable només funciona amb el servidor nou", 0)
                ],
                "required": [
                    ("versió estable", 0),
                    ("només funciona", 0),
                    ("servidor nou", 0),
                ],
                "allowed": ("La versió estable només funciona amb el servidor nou.", 0),
                "notes": "The restriction belongs to the stable version, not the test version.",
            }
        ],
    )
    source = "العميل طلب النسخة العربية. التسليم يوم الثلاثاء، وليس يوم الاثنين."
    add_authored(
        "arabic_delivery_correction",
        "arabic",
        source,
        [
            {
                "canonical_text": "التسليم يوم الثلاثاء وليس يوم الاثنين.",
                "expected": [("التسليم يوم الثلاثاء، وليس يوم الاثنين", 0)],
                "required": [("التسليم يوم الثلاثاء", 0), ("وليس يوم الاثنين", 0)],
                "allowed": ("التسليم يوم الثلاثاء، وليس يوم الاثنين.", 0),
                "notes": "The corrected Tuesday date and rejected Monday date should remain together.",
            }
        ],
    )
    source = "校对完成后，报告交给林。预算表仍由陈保管。"
    add_authored(
        "cjk_distinct_owners",
        "cjk",
        source,
        [
            {
                "canonical_text": "校对完成后，报告交给林。",
                "expected": [("校对完成后，报告交给林", 0)],
                "required": [("报告交给林", 0)],
                "allowed": ("校对完成后，报告交给林。", 0),
                "notes": "CJK characters use individually visible anchors; the report recipient is Lin.",
            },
            {
                "canonical_text": "预算表由陈保管。",
                "expected": [("预算表仍由陈保管", 0)],
                "required": [("预算表仍由陈保管", 0)],
                "allowed": ("预算表仍由陈保管。", 0),
                "notes": "The budget-sheet custodian is Chen, separately from the report.",
            },
        ],
    )
    source = "Build note: set `cache.ttl_ms=750`; the previous value was 1500.\r\nDeploy note: restart the read worker after changing it."
    add_authored(
        "code_crlf",
        "code_punctuation_crlf",
        source,
        [
            {
                "canonical_text": "The new cache.ttl_ms value is 750.",
                "expected": [("set `cache.ttl_ms=750`", 0)],
                "required": [("cache.ttl_ms=750", 0)],
                "allowed": (
                    "Build note: set `cache.ttl_ms=750`; the previous value was 1500.",
                    0,
                ),
                "notes": "The code-like setting must retain exact punctuation; the CRLF is in the source snapshot.",
            }
        ],
    )
    source = "The north gate closes at 18:00. Staff with an evening badge can still enter until 20:00."
    add_authored(
        "adjacent_clause_exception",
        "two_clauses",
        source,
        [
            {
                "canonical_text": "The north gate closes at 18:00, but staff with evening badges may enter until 20:00.",
                "expected": [(source, 0)],
                "required": [
                    ("north gate closes at 18:00", 0),
                    ("Staff with an evening badge can still enter until 20:00", 0),
                ],
                "allowed": (source, 0),
                "notes": "Both adjacent clauses are needed to support the exception; neither clause alone is enough.",
            }
        ],
    )
    source = "Stock count: 42 units arrived.\nAfter inspection, six damaged units were rejected."
    add_authored(
        "two_clause_net_count",
        "two_clauses",
        source,
        [
            {
                "canonical_text": "Thirty-six accepted units remained after inspection.",
                "expected": [(source, 0)],
                "required": [
                    ("42 units arrived", 0),
                    ("six damaged units were rejected", 0),
                ],
                "allowed": (source, 0),
                "notes": "Both counts support the simple arithmetic; the candidate must not cite either alone.",
            }
        ],
    )
    source = "The user asked whether the museum opens on Sunday. The reply only listed weekday hours."
    add_authored(
        "question_not_fact",
        "unsupported",
        source,
        [
            {
                "canonical_text": "The museum opens on Sunday.",
                "expected": None,
                "notes": "A question about Sunday is not evidence of Sunday opening.",
            }
        ],
    )
    source = "El martes no puedo asistir; el jueves sí tengo disponibilidad para la revisión."
    add_authored(
        "spanish_negation",
        "spanish",
        source,
        [
            {
                "canonical_text": "Puede asistir a la revisión el jueves, pero no el martes.",
                "expected": [(source, 0)],
                "required": [
                    ("martes no puedo asistir", 0),
                    ("jueves sí tengo disponibilidad", 0),
                ],
                "allowed": (source, 0),
                "notes": "Both the negative Tuesday and positive Thursday statements are essential.",
            }
        ],
    )
    source = "Please record that the permit is valid through 30 April, subject to the inspector's signature."
    add_authored(
        "conditional_validity",
        "qualification",
        source,
        [
            {
                "canonical_text": "The permit is valid through 30 April if the inspector signs it.",
                "expected": [
                    (
                        "permit is valid through 30 April, subject to the inspector's signature",
                        0,
                    )
                ],
                "required": [
                    ("through 30 April", 0),
                    ("subject to the inspector's signature", 0),
                ],
                "allowed": (source, 0),
                "notes": "The signature condition cannot be dropped from the supporting range.",
            }
        ],
    )
    source = (
        "The service is paused, not deleted. Its data remains available for export."
    )
    add_authored(
        "status_not_deletion",
        "negation",
        source,
        [
            {
                "canonical_text": "The service is paused and its data can be exported.",
                "expected": [(source, 0)],
                "required": [
                    ("paused, not deleted", 0),
                    ("data remains available for export", 0),
                ],
                "allowed": (source, 0),
                "notes": "The paused state and export availability are in adjacent sentences.",
            },
            {
                "canonical_text": "The service and its data were deleted.",
                "expected": None,
                "notes": "The source explicitly rejects deletion.",
            },
        ],
    )
    source = "The revised route is /v2/accounts/{account_id}/events?limit=20. Do not use /v1/accounts for this report."
    add_authored(
        "path_and_query",
        "code_punctuation",
        source,
        [
            {
                "canonical_text": "The report uses /v2/accounts/{account_id}/events?limit=20.",
                "expected": [
                    ("revised route is /v2/accounts/{account_id}/events?limit=20", 0)
                ],
                "required": [("/v2/accounts/{account_id}/events?limit=20", 0)],
                "allowed": (
                    "The revised route is /v2/accounts/{account_id}/events?limit=20.",
                    0,
                ),
                "notes": "The complete path, braces, query key, and value are necessary.",
            }
        ],
    )

    long_specs = (
        (
            "long_300",
            19,
            16,
            "Release note: the orange batch is scheduled for Tuesday, after the review call.",
            "orange batch",
            "scheduled for Tuesday",
        ),
        (
            "long_600",
            37,
            33,
            "Handover note: the signed map stays with the depot manager until Friday.",
            "signed map",
            "until Friday",
        ),
        (
            "long_1100",
            72,
            67,
            "Final routing note: the public notice goes to the west office only after legal approval.",
            "public notice",
            "only after legal approval",
        ),
    )
    for case_id, sections, insert_after, evidence, subject, condition in long_specs:
        source = _long_document(sections, insert_after, evidence)
        catalog = SourceReferenceCatalog(source)
        minimum = {"long_300": 255, "long_600": 500, "long_1100": 1000}[case_id]
        if len(catalog.anchors) <= minimum:
            raise ValueError(f"{case_id} needs more than {minimum} visible anchors")
        cases.append(
            _case(
                case_id,
                "length_stressed",
                source,
                [
                    _candidate(
                        source,
                        f"{subject} {condition}.",
                        expected=[(evidence, 0)],
                        required=[(subject, 0), (condition, 0)],
                        allowed=(evidence, 0),
                        notes="Relevant evidence appears near the end of a varied multi-section operational report; citing the whole report is too broad.",
                    )
                ],
                kind="length_stressed",
                origin_notes=f"Authored varied operational sections; {len(catalog.anchors)} visible production anchors. No holdout source.",
            )
        )

    if not 24 <= len(cases) <= 32:
        raise ValueError(f"Unexpected fixture count: {len(cases)}")
    return cases
