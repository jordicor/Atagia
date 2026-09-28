"""Frozen, compact-selection cases with several candidates per source message.

Metadata expectations are diagnostics separate from quote selection scoring.
All messages are newly authored synthetic development material.
"""

from __future__ import annotations

from atagia.core.source_references import SourceReferenceCatalog

from benchmarks.source_quote_selection.acceptance_cases import _candidate, _case


def _add_case(
    cases: list[dict],
    case_id: str,
    source: str,
    specifications: list[dict],
    *,
    recent_messages: list[dict] | None = None,
    prior_chunk_context: str | None = None,
    note: str,
) -> None:
    candidates = []
    metadata_expectations = {}
    for specification in specifications:
        fields = specification.copy()
        metadata = fields.pop("metadata", None)
        candidate = _candidate(source, **fields)
        candidates.append(candidate)
        if metadata is not None:
            metadata_expectations[candidate["candidate_id"]] = metadata
    case = _case(
        case_id,
        source,
        candidates,
        recent_messages=recent_messages,
        prior_chunk_context=prior_chunk_context,
        note=note,
    )
    case["metadata_expectations"] = metadata_expectations
    cases.append(case)


def _metadata(
    support_kind: str, language_codes: list[str], preserve_verbatim: bool = False
) -> dict:
    return {
        "support_kind": support_kind,
        "language_codes": language_codes,
        "preserve_verbatim": preserve_verbatim,
    }


def _anchor_bounds(source: str, interval: list[int]) -> tuple[int, int]:
    anchors = SourceReferenceCatalog(source).anchors
    first = next(
        index
        for index, anchor in enumerate(anchors, 1)
        if anchor.char_start == interval[0]
    )
    last = next(
        index
        for index, anchor in enumerate(anchors, 1)
        if anchor.char_end == interval[1]
    )
    return first, last


def load_cases() -> list[dict]:
    """Return eight new JSON-serializable cases, without evaluation or API calls."""
    cases: list[dict] = []

    source = (
        "The Iris release is tagged `v2.4.1`. "
        "I approved deployment to staging, not production. "
        "Ana owns the rollback checklist. "
        "The database backup starts at 02:15 UTC. "
        "Keep the old dashboard link active until the audit ends. "
        "Finance has not approved the extra server. "
        "The incident report is due Friday."
    )
    _add_case(
        cases,
        "compact_release_handoff",
        source,
        [
            dict(
                candidate_id="cand_001",
                canonical_text="The Iris release tag is v2.4.1.",
                support_kind="direct",
                exact=(
                    "The Iris release is tagged `v2.4.1`",
                    "The Iris release is tagged `v2.4.1`.",
                ),
                required=("`v2.4.1`",),
                allowed="The Iris release is tagged `v2.4.1`.",
                note="Terminal period is optional; backticks and dots inside the tag remain mandatory.",
                metadata=_metadata("direct", ["en"], True),
            ),
            dict(
                candidate_id="cand_002",
                canonical_text="Deployment was approved for staging, not production.",
                support_kind="direct",
                exact=(
                    "I approved deployment to staging, not production",
                    "I approved deployment to staging, not production.",
                ),
                required=("staging, not production",),
                allowed="I approved deployment to staging, not production.",
                note="The production exclusion must remain with the staging approval; final period is optional.",
                metadata=_metadata("direct", ["en"]),
            ),
            dict(
                candidate_id="cand_003",
                canonical_text="Ana owns the rollback checklist.",
                support_kind="direct",
                exact=(
                    "Ana owns the rollback checklist",
                    "Ana owns the rollback checklist.",
                ),
                required=("Ana owns", "rollback checklist"),
                allowed="Ana owns the rollback checklist.",
                note="Specific owner and duty are required; final period is optional.",
                metadata=_metadata("direct", ["en"]),
            ),
            dict(
                candidate_id="cand_004",
                canonical_text="The database backup starts at 02:15 UTC.",
                support_kind="direct",
                exact=(
                    "The database backup starts at 02:15 UTC",
                    "The database backup starts at 02:15 UTC.",
                ),
                required=("database backup", "02:15 UTC"),
                allowed="The database backup starts at 02:15 UTC.",
                note="The colon and digits in the time are mandatory; final period is optional.",
                metadata=_metadata("direct", ["en"], True),
            ),
            dict(
                candidate_id="cand_005",
                canonical_text="The old dashboard link should stay active until the audit ends.",
                support_kind="direct",
                exact=(
                    "Keep the old dashboard link active until the audit ends",
                    "Keep the old dashboard link active until the audit ends.",
                ),
                required=("old dashboard link", "until the audit ends"),
                allowed="Keep the old dashboard link active until the audit ends.",
                note="The audit boundary is essential; final period is optional.",
                metadata=_metadata("direct", ["en"]),
            ),
            dict(
                candidate_id="cand_006",
                canonical_text="Finance has not approved the extra server.",
                support_kind="direct",
                exact=(
                    "Finance has not approved the extra server",
                    "Finance has not approved the extra server.",
                ),
                required=("Finance has not approved", "extra server"),
                allowed="Finance has not approved the extra server.",
                note="Negation and approver attribution are mandatory; final period is optional.",
                metadata=_metadata("direct", ["en"]),
            ),
            dict(
                candidate_id="cand_007",
                canonical_text="The incident report is due Friday.",
                support_kind="direct",
                exact=(
                    "The incident report is due Friday",
                    "The incident report is due Friday.",
                ),
                required=("incident report", "due Friday"),
                allowed="The incident report is due Friday.",
                note="Report and date must both appear; final period is optional.",
                metadata=_metadata("direct", ["en"]),
            ),
            dict(
                candidate_id="cand_008",
                canonical_text="Production deployment was approved.",
                support_kind="unsupported",
                note="The source explicitly withholds production approval.",
            ),
        ],
        note="Eight compact candidates test exact values, ownership, timing, qualifications, and unsupported production approval.",
    )

    source = (
        "The second one, provided the loading desk confirms by noon. "
        "Ask Noor to bring the green folder, not the blue copy. "
        "Put the receipt in the side pocket. "
        "I will call the driver after lunch."
    )
    _add_case(
        cases,
        "compact_elliptical_pickup",
        source,
        [
            dict(
                candidate_id="cand_001",
                canonical_text="Choose the Wednesday 16:00 pickup if the loading desk confirms by noon.",
                support_kind="contextual",
                exact=(
                    "The second one, provided the loading desk confirms by noon",
                    "The second one, provided the loading desk confirms by noon.",
                ),
                required=("second one", "provided the loading desk confirms by noon"),
                allowed="The second one, provided the loading desk confirms by noon.",
                note="Recent options identify the second slot; the confirmation condition cannot be dropped. Final period is optional.",
                metadata=_metadata("contextual_direct", ["en"]),
            ),
            dict(
                candidate_id="cand_002",
                canonical_text="The user wants Noor asked to bring the green folder, not the blue copy.",
                support_kind="direct",
                exact=(
                    "Ask Noor to bring the green folder, not the blue copy",
                    "Ask Noor to bring the green folder, not the blue copy.",
                ),
                required=("Ask Noor", "green folder, not the blue copy"),
                allowed="Ask Noor to bring the green folder, not the blue copy.",
                note="An instruction to ask Noor is not a claim that Noor has already agreed; final period is optional.",
                metadata=_metadata("direct", ["en"]),
            ),
            dict(
                candidate_id="cand_003",
                canonical_text="The receipt should go in the side pocket.",
                support_kind="direct",
                exact=(
                    "Put the receipt in the side pocket",
                    "Put the receipt in the side pocket.",
                ),
                required=("receipt", "side pocket"),
                allowed="Put the receipt in the side pocket.",
                note="Keep the object and location together; final period is optional.",
                metadata=_metadata("direct", ["en"]),
            ),
            dict(
                candidate_id="cand_004",
                canonical_text="The user will call the driver after lunch.",
                support_kind="direct",
                exact=(
                    "I will call the driver after lunch",
                    "I will call the driver after lunch.",
                ),
                required=("I will call the driver", "after lunch"),
                allowed="I will call the driver after lunch.",
                note="Speaker and timing belong to the same commitment; final period is optional.",
                metadata=_metadata("direct", ["en"]),
            ),
            dict(
                candidate_id="cand_005",
                canonical_text="The loading desk has already confirmed the pickup.",
                support_kind="unsupported",
                note="Confirmation is a future condition, not a completed fact.",
            ),
        ],
        recent_messages=[
            {
                "role": "assistant",
                "content": "Pickup options are Tuesday at 10:00 or Wednesday at 16:00. Which slot should I request?",
            }
        ],
        note="A short answer inherits an option from recent conversation while distinct instructions remain local to their own clauses.",
    )

    source = (
        "Para el taller, reserva la sala Norte el jueves. "
        "Marta dijo que la llave está lista, pero yo no la he probado. "
        "El código de acceso es `N-42/7`. "
        "No envíes el acta hasta que firme Lucía. "
        "Si no llega la firma, dejamos la reunión en borrador."
    )
    _add_case(
        cases,
        "compact_spanish_workshop",
        source,
        [
            dict(
                candidate_id="cand_001",
                canonical_text="El taller se reserva en la sala Norte el jueves.",
                support_kind="direct",
                exact=(
                    "Para el taller, reserva la sala Norte el jueves",
                    "Para el taller, reserva la sala Norte el jueves.",
                ),
                required=("sala Norte", "jueves"),
                allowed="Para el taller, reserva la sala Norte el jueves.",
                note="Room and day are mandatory; final period is optional.",
                metadata=_metadata("direct", ["es"]),
            ),
            dict(
                candidate_id="cand_002",
                canonical_text="Marta dice que la llave está lista, pero la persona usuaria aún no la ha probado.",
                support_kind="direct",
                exact=(
                    "Marta dijo que la llave está lista, pero yo no la he probado",
                    "Marta dijo que la llave está lista, pero yo no la he probado.",
                ),
                required=("Marta dijo", "yo no la he probado"),
                allowed="Marta dijo que la llave está lista, pero yo no la he probado.",
                note="Attribution and untested qualification must both appear; final period is optional.",
                metadata=_metadata("direct", ["es"]),
            ),
            dict(
                candidate_id="cand_003",
                canonical_text="El código de acceso es N-42/7.",
                support_kind="direct",
                exact=(
                    "El código de acceso es `N-42/7`",
                    "El código de acceso es `N-42/7`.",
                ),
                required=("`N-42/7`",),
                allowed="El código de acceso es `N-42/7`.",
                note="Hyphen, slash, digits, and backticks belong to the exact value; final period is optional.",
                metadata=_metadata("direct", ["es"], True),
            ),
            dict(
                candidate_id="cand_004",
                canonical_text="El acta no debe enviarse hasta que firme Lucía.",
                support_kind="direct",
                exact=(
                    "No envíes el acta hasta que firme Lucía",
                    "No envíes el acta hasta que firme Lucía.",
                ),
                required=("No envíes", "hasta que firme Lucía"),
                allowed="No envíes el acta hasta que firme Lucía.",
                note="The negative instruction and signature condition must stay together; final period is optional.",
                metadata=_metadata("direct", ["es"]),
            ),
            dict(
                candidate_id="cand_005",
                canonical_text="Sin la firma, la reunión queda en borrador.",
                support_kind="direct",
                exact=(
                    "Si no llega la firma, dejamos la reunión en borrador",
                    "Si no llega la firma, dejamos la reunión en borrador.",
                ),
                required=("Si no llega la firma", "en borrador"),
                allowed="Si no llega la firma, dejamos la reunión en borrador.",
                note="Conditional status is not an unconditional outcome; final period is optional.",
                metadata=_metadata("direct", ["es"]),
            ),
            dict(
                candidate_id="cand_006",
                canonical_text="La persona usuaria ha probado la llave y confirmó que funciona.",
                support_kind="unsupported",
                note="The user explicitly says the key has not been tested.",
            ),
        ],
        note="Spanish clauses distinguish speaker reports, personal verification, exact access code, and conditional sending.",
    )

    source = (
        "周三把蓝色封套交给王，不要交给李。\r\n"
        "发票编号是 `CN-08/4`，金额为 320 元。\r\n"
        "会议改到下午三点，地点仍是西楼。"
    )
    _add_case(
        cases,
        "compact_chinese_crlf",
        source,
        [
            dict(
                candidate_id="cand_001",
                canonical_text="周三把蓝色封套交给王，不交给李。",
                support_kind="direct",
                exact=(
                    "周三把蓝色封套交给王，不要交给李",
                    "周三把蓝色封套交给王，不要交给李。",
                ),
                required=("周三", "交给王", "不要交给李"),
                allowed="周三把蓝色封套交给王，不要交给李。",
                note="Day, intended recipient, and excluded recipient are all necessary; final period is optional.",
                metadata=_metadata("direct", ["zh"]),
            ),
            dict(
                candidate_id="cand_002",
                canonical_text="发票编号是 CN-08/4。",
                support_kind="direct",
                exact=("发票编号是 `CN-08/4`", "发票编号是 `CN-08/4`，"),
                required=("`CN-08/4`",),
                allowed="发票编号是 `CN-08/4`，",
                note="The clause comma is optional; punctuation inside the invoice number is mandatory.",
                metadata=_metadata("direct", ["zh"], True),
            ),
            dict(
                candidate_id="cand_003",
                canonical_text="发票金额为 320 元。",
                support_kind="direct",
                exact=("金额为 320 元", "金额为 320 元。"),
                required=("320 元",),
                allowed="金额为 320 元。",
                note="The amount and currency stay together; final period is optional.",
                metadata=_metadata("direct", ["zh"], True),
            ),
            dict(
                candidate_id="cand_004",
                canonical_text="会议改到下午三点，地点仍是西楼。",
                support_kind="direct",
                exact=(
                    "会议改到下午三点，地点仍是西楼",
                    "会议改到下午三点，地点仍是西楼。",
                ),
                required=("改到下午三点", "仍是西楼"),
                allowed="会议改到下午三点，地点仍是西楼。",
                note="Changed time and unchanged venue are both required; final period is optional.",
                metadata=_metadata("direct", ["zh"]),
            ),
            dict(
                candidate_id="cand_005",
                canonical_text="蓝色封套应交给李。",
                support_kind="unsupported",
                note="The source expressly excludes Li as recipient.",
            ),
        ],
        note="Chinese text and CRLF test independent facts, literal invoice signs, and negative recipient attribution.",
    )

    migration_lines = [
        "The migration team opened the morning review by checking the test tenant's data inventory against yesterday's export manifest and recording three missing attachment labels.",
        "Rina traced the labels to an old archive job, while Omar confirmed that the underlying files remained readable in the local staging copy.",
        "The first run finished before lunch, but the scheduler kept its write lock until a worker completed the verification summary and released the queue.",
        "Operations moved the next rehearsal into a reserved window so the support desk could answer customer tickets without competing for the same database connection.",
        "The finance observer compared item totals across two reports, found a rounding difference, and asked the team to preserve the raw ledger export for review.",
        "During the afternoon check, the identity owner verified that temporary credentials reached only the named migration operators and that each operator's access had an expiry.",
        "The team paused to document how archived attachments map to their new object identifiers, since an earlier draft had omitted the linkage field.",
        "The storage engineer sampled objects from old and new buckets, matching their checksums before proposing another rehearsal with a larger but still synthetic dataset.",
        "A support analyst checked the help-center article and found that the rollback step mentioned an obsolete button label, which documentation corrected that evening.",
        "Approval arrived for the migration after the afternoon check. The approval applies only to the test tenant, so production remains blocked.",
        "After the approval discussion, the release coordinator logged the tenant boundary in the change record and sent the revised steps to the two reviewers.",
        "The reporting service produced a fresh summary after the rehearsal; its event count matched the accepted import count but not the earlier provisional count.",
        "At the next check-in, Omar agreed to watch the error queue for thirty minutes while Rina ran the permission audit and recorded its exact finish time.",
        "The team left the old read path available for the test tenant until the signed verification checklist and the new lookup results were both reviewed.",
        "For the overnight plan, operations reserved a short maintenance window and assigned a separate contact to handle any urgent customer escalation.",
        "The final handoff lists the archive mapping, the signed checklist, the report totals, and the remaining production approval as four separate review items.",
        "No production switch was scheduled; the decision log says a later owner review will set that date after the test tenant's open issues close.",
    ]
    source = "\n".join(migration_lines)
    _add_case(
        cases,
        "long_migration_boundary",
        source,
        [
            dict(
                candidate_id="cand_001",
                canonical_text="The archive labels were missing, but the underlying files remained readable in staging.",
                support_kind="direct",
                exact=(migration_lines[0] + "\n" + migration_lines[1],),
                required=(
                    "three missing attachment labels",
                    "underlying files remained readable",
                ),
                allowed=migration_lines[0] + "\n" + migration_lines[1],
                note="Adjacent early updates distinguish missing labels from inaccessible files; the newline is part of the exact quote.",
                metadata=_metadata("direct", ["en"]),
            ),
            dict(
                candidate_id="cand_002",
                canonical_text="Migration approval covers only the test tenant; production remains blocked.",
                support_kind="direct",
                exact=(migration_lines[9],),
                required=(
                    "Approval arrived for the migration after the afternoon check",
                    "only to the test tenant",
                    "production remains blocked",
                ),
                allowed=migration_lines[9],
                note="Both sentences form one sufficient passage; the test-tenant restriction and production block are mandatory.",
                metadata=_metadata("direct", ["en"]),
            ),
            dict(
                candidate_id="cand_003",
                canonical_text="A later owner review will set the production switch date after test issues close.",
                support_kind="direct",
                exact=(migration_lines[-1],),
                required=(
                    "No production switch was scheduled",
                    "later owner review will set that date",
                    "after the test tenant's open issues close",
                ),
                allowed=migration_lines[-1],
                note="The late correction denies any present date and states the conditional next step.",
                metadata=_metadata("direct", ["en"]),
            ),
            dict(
                candidate_id="cand_004",
                canonical_text="Production migration was approved for tonight.",
                support_kind="unsupported",
                note="Approval is test-tenant-only and no production switch was scheduled.",
            ),
        ],
        note="Distributed early, boundary-crossing, and late evidence in a coherent migration handoff.",
    )

    exhibit_lines = [
        "The exhibition committee first reviewed the room map and asked for a clear visitor route between the ceramics gallery, the reading alcove, and the courtyard entrance.",
        "Conservation documented the condition of each loaned object and photographed the packing materials so the museum could reverse any later handling decision.",
        "A volunteer coordinator counted the available guides for Saturday and assigned the morning group to the east entrance where the school buses arrive.",
        "The accessibility lead checked turning space near the display cases, then requested a wider path around the central table before approving the final map.",
        "The design studio printed two label sizes and tested both with visitors who were unfamiliar with the collection's specialized vocabulary.",
        "The registrar compared loan forms with the crate inventory and flagged a missing signature on one watercolor's condition report for follow-up.",
        "At the daily meeting, staff decided to move the fragile paper works away from the bright window and to monitor light exposure during installation.",
        "The education team wrote a short workshop outline that introduces the materials through touch samples rather than letting children handle the loaned works.",
        "Facilities checked the courtyard canopy and measured the distance from the temporary power outlet to the planned outdoor projection screen.",
        "The museum shop asked whether the catalog could arrive before opening weekend, but the printer had only promised a proof for Thursday.",
        "The curator said the bronze figure could travel with the first shipment; the lender later corrected that only the pedestal was cleared for transport.",
        "After that correction, the packing team left the bronze figure in secure storage and prepared a separate inventory line for the pedestal.",
        "The communications office drafted an opening notice but held publication until the lender approved the final artwork credit and image crop.",
        "The registrar called the lender again, recorded the approval for the credit line, and forwarded the signed note to the communications owner.",
        "The installation lead reserved two lifts for Friday morning so the heavy display frames could be positioned before the gallery floor reopened.",
        "A safety inspection found a loose cable cover along the visitor route and required repair before any public preview could begin.",
        "The final planning note keeps the bronze figure in storage, sends the pedestal first, and schedules a new transport decision after the lender's conservation review.",
    ]
    source = "\n".join(exhibit_lines)
    _add_case(
        cases,
        "long_exhibition_correction",
        source,
        [
            dict(
                candidate_id="cand_001",
                canonical_text="The gallery route needs a wider path around the central table.",
                support_kind="direct",
                exact=(exhibit_lines[3],),
                required=("requested a wider path around the central table",),
                allowed=exhibit_lines[3],
                note="This early accessibility request is local to the route map.",
                metadata=_metadata("direct", ["en"]),
            ),
            dict(
                candidate_id="cand_002",
                canonical_text="Only the pedestal was cleared for transport; the bronze figure remains in storage.",
                support_kind="direct",
                exact=(exhibit_lines[10] + "\n" + exhibit_lines[11],),
                required=(
                    "only the pedestal was cleared for transport",
                    "left the bronze figure in secure storage",
                ),
                allowed=exhibit_lines[10] + "\n" + exhibit_lines[11],
                note="The lender's later correction overrides the curator's initial statement; both adjacent lines are needed.",
                metadata=_metadata("direct", ["en"]),
            ),
            dict(
                candidate_id="cand_003",
                canonical_text="A public preview must wait until the loose cable cover is repaired.",
                support_kind="direct",
                exact=(exhibit_lines[-2],),
                required=(
                    "loose cable cover",
                    "required repair before any public preview",
                ),
                allowed=exhibit_lines[-2],
                note="A late safety condition must not become an unconditional preview date.",
                metadata=_metadata("direct", ["en"]),
            ),
            dict(
                candidate_id="cand_004",
                canonical_text="The bronze figure shipped with the first shipment.",
                support_kind="unsupported",
                note="The curator's initial belief was explicitly corrected by the lender.",
            ),
        ],
        note="A long exhibition log separates early accessibility, corrected lender authority, and late safety conditions.",
    )

    release_lines = [
        "The engineering group began with a review of last week's incident notes and assigned owners for three alerts that still lacked clear recovery instructions.\r",
        "The frontend team checked the revised account page with keyboard navigation and logged two focus-order problems for the accessibility review.\r",
        "Infrastructure measured request latency under a synthetic read load and found the database remained stable while a logging worker used more CPU than expected.\r",
        "The release manager froze the staging checklist after each component owner confirmed its build artifact and the matching source revision.\r",
        "One operator noticed that an internal dashboard still displayed the old service name, so documentation updated the screenshot before the training session.\r",
        "The data group compared record counts before and after its replay and investigated nine rows whose timestamps differed because of an import timezone setting.\r",
        "A security reviewer verified that the test token could read sample reports but could not edit production account settings.\r",
        "The API note sets `quota/day=1,250` for sandbox traffic; the production quota remains unchanged.\r",
        "The integration team generated a new sample event and confirmed that its schema version matched the public contract for the pending release.\r",
        "Support rehearsed the escalation path with the on-call engineer and updated a phone tree whose second contact had left the team.\r",
        "The build pipeline produced a signed package, then stored its checksum beside the release manifest for the eventual deployment review.\r",
        "The monitoring owner tuned one staging alert to reduce duplicate notices while keeping a separate high-severity threshold untouched.\r",
        "The QA lead reported that all ordinary account flows passed, but the export test still failed when filenames contained a percent sign.\r",
        "Because the export defect remained open, the release manager said the production rollout was postponed until a fixed build passes that test.\r",
        "The team recorded the exact export filename and trace identifier in a private issue, then shared a sanitized reproduction with the component owner.\r",
        "At the closing review, operations checked that the rollback package could still be loaded from the artifact store without relying on the staging host.\r",
        "The final note assigns the export correction to Priya, asks QA to rerun that single case, and leaves the rollout date unset.\r",
    ]
    source = "\n".join(release_lines)
    _add_case(
        cases,
        "long_release_crlf",
        source,
        [
            dict(
                candidate_id="cand_001",
                canonical_text="The sandbox quota is quota/day=1,250; the production quota is unchanged.",
                support_kind="direct",
                exact=(release_lines[7].rstrip("\r"),),
                required=("`quota/day=1,250`", "production quota remains unchanged"),
                allowed=release_lines[7].rstrip("\r"),
                note="Slash, equals sign, comma, digits, and backticks are mandatory; the source uses CRLF line endings.",
                metadata=_metadata("direct", ["en"], True),
            ),
            dict(
                candidate_id="cand_002",
                canonical_text="The production rollout is postponed until the export test passes with a fixed build.",
                support_kind="direct",
                exact=(
                    release_lines[12].rstrip("\r")
                    + "\r\n"
                    + release_lines[13].rstrip("\r"),
                ),
                required=(
                    "export test still failed",
                    "production rollout was postponed until a fixed build passes that test",
                ),
                allowed=release_lines[12].rstrip("\r")
                + "\r\n"
                + release_lines[13].rstrip("\r"),
                note="The failure and consequent postponement require adjacent CRLF-separated lines.",
                metadata=_metadata("direct", ["en"]),
            ),
            dict(
                candidate_id="cand_003",
                canonical_text="Priya owns the export correction and the rollout date remains unset.",
                support_kind="direct",
                exact=(release_lines[-1].rstrip("\r"),),
                required=("export correction to Priya", "rollout date unset"),
                allowed=release_lines[-1].rstrip("\r"),
                note="Ownership and unset date appear together in the final note.",
                metadata=_metadata("direct", ["en"]),
            ),
            dict(
                candidate_id="cand_004",
                canonical_text="The production rollout is approved for tomorrow.",
                support_kind="unsupported",
                note="The rollout was postponed and its date left unset.",
            ),
        ],
        note="A CRLF release report distributes exact config, a two-line causal decision, and late ownership.",
    )

    festival_lines = [
        "The neighborhood festival committee met on Monday to check the permit conditions, the volunteer roster, and the delivery route for temporary tables.",
        "The site coordinator walked the square with a city inspector and marked two narrow turns where the accessible route needs a portable ramp.",
        "A food vendor asked for an earlier setup window, but the organizer explained that the service gate stays closed until the morning safety inspection.",
        "The finance volunteer compared the stall deposits with the signed vendor list and found one payment whose reference number was missing a digit.",
        "The sound engineer tested the small stage at rehearsal volume and asked performers to keep monitors pointed away from the residential windows.",
        "The library agreed to lend folding chairs for the reading tent, provided the committee returns them before the next day's school program.",
        "The transportation lead reserved a van for the chairs and identified a second driver in case the first volunteer had to leave early.",
        "At the permit call, the city officer said amplified music could continue until 21:00, while the organizer's draft poster still said 22:00.",
        "The communications owner corrected the poster time and sent a revised proof to the city officer before scheduling any public post.",
        "A parent group proposed a craft table near the fountain, but the safety marshal asked them to move it away from the wet stone edge.",
        "The weather coordinator tracked two forecasts and advised the team to keep the indoor reading room available if afternoon rain arrived.",
        "The first group of volunteers practiced the queue route and discovered that one sign pointed people toward a locked side door.",
        "Facilities replaced that sign, then checked that visitors could find drinking water without crossing the stage equipment area.",
        "The organizer asked the caterer for a written ingredient sheet before assigning food labels, since verbal assurances were not enough for the allergy notice.",
        "The arts coordinator confirmed that the mural workshop has washable materials and that the venue will provide a separate cleanup sink.",
        "The city officer returned the revised proof with approval for the corrected 21:00 music cutoff, not for any later extension.",
        "The volunteer lead sent a final roster showing who opens the reading tent, who checks the ramp, and who closes the service gate.",
        "The closing plan says the poster may go live after the city approval, while the ingredient sheet remains an open prerequisite for food labels.",
    ]
    source = "\n".join(festival_lines)
    _add_case(
        cases,
        "long_festival_permit",
        source,
        [
            dict(
                candidate_id="cand_001",
                canonical_text="The library will lend chairs if they are returned before the next day's school program.",
                support_kind="direct",
                exact=(festival_lines[5],),
                required=(
                    "library agreed to lend folding chairs",
                    "provided the committee returns them before the next day's school program",
                ),
                allowed=festival_lines[5],
                note="The lending condition is essential and occurs early in the message.",
                metadata=_metadata("direct", ["en"]),
            ),
            dict(
                candidate_id="cand_002",
                canonical_text="Amplified music must stop at 21:00; the organizer's 22:00 poster was wrong.",
                support_kind="direct",
                exact=(festival_lines[7] + "\n" + festival_lines[8],),
                required=(
                    "continue until 21:00",
                    "draft poster still said 22:00",
                    "corrected the poster time",
                ),
                allowed=festival_lines[7] + "\n" + festival_lines[8],
                note="The authoritative city time and correction must be cited together; colons and digits are mandatory.",
                metadata=_metadata("direct", ["en"], True),
            ),
            dict(
                candidate_id="cand_003",
                canonical_text="Food labels still require a written ingredient sheet.",
                support_kind="direct",
                exact=(festival_lines[-1],),
                required=(
                    "ingredient sheet remains an open prerequisite for food labels",
                ),
                allowed=festival_lines[-1],
                note="The late open prerequisite is distinct from poster approval.",
                metadata=_metadata("direct", ["en"]),
            ),
            dict(
                candidate_id="cand_004",
                canonical_text="The city approved music until 22:00.",
                support_kind="unsupported",
                note="The city explicitly approved the corrected 21:00 cutoff and no later extension.",
            ),
        ],
        note="A long event plan tests official-versus-draft attribution, corrected numeric cutoff, and separate pending condition.",
    )

    if len(cases) != 8:
        raise ValueError("The compact corpus must contain exactly eight cases")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Compact case IDs must be unique")
    for case in cases[:4]:
        if not 4 <= len(case["candidates"]) <= 8:
            raise ValueError(f"Invalid compact candidate count: {case['case_id']}")
        if len(SourceReferenceCatalog(case["source_text"]).anchors) > 100:
            raise ValueError(f"Short source has too many anchors: {case['case_id']}")
    for case in cases[4:]:
        count = len(SourceReferenceCatalog(case["source_text"]).anchors)
        if not 400 <= count <= 700:
            raise ValueError(
                f"Long source outside anchor range: {case['case_id']}, {count}"
            )
        if (
            sum(bool(candidate["expected_ranges"]) for candidate in case["candidates"])
            > 3
        ):
            raise ValueError(
                f"Long source has too many supported candidates: {case['case_id']}"
            )
    boundary_case = cases[4]
    first, last = _anchor_bounds(
        boundary_case["source_text"],
        boundary_case["candidates"][1]["expected_ranges"][0],
    )
    if not first <= 254 < last:
        raise ValueError(
            f"Migration qualification must cross an anchor block: {first}-{last}"
        )
    return cases
