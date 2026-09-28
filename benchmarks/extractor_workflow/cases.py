"""Fixed synthetic cases for blind manual review of the full extractor.

Facts are semantic review labels, not string-matching rules. Kind, scope,
temporal, and verbatim labels are diagnostics separate from fact recall.
Exact source coordinates should be checked against the production catalog.
"""

from __future__ import annotations

import json


def _fact(
    meaning: str,
    *,
    kind: str | None = None,
    scope: str | None = None,
    temporal_type: str | None = None,
    preserve_verbatim: bool | None = None,
) -> dict:
    result: dict[str, object] = {"meaning": meaning}
    if kind is not None:
        result["kind"] = kind
    if scope is not None:
        result["scope"] = scope
    if temporal_type is not None:
        result["temporal_type"] = temporal_type
    if preserve_verbatim is not None:
        result["preserve_verbatim"] = preserve_verbatim
    return result


def _case(
    case_id: str,
    source_text: str,
    *,
    required_facts: list[dict],
    forbidden_claims: list[str],
    optional_facts: list[dict] | None = None,
    recent_messages: list[dict] | None = None,
    prior_chunk_context: str | None = None,
    role: str = "user",
    note: str,
) -> dict:
    return {
        "case_id": case_id,
        "source_text": source_text,
        "role": role,
        "recent_messages": recent_messages or [],
        "prior_chunk_context": prior_chunk_context,
        "required_facts": required_facts,
        "forbidden_claims": forbidden_claims,
        "optional_facts": optional_facts or [],
        "expected_nothing_durable": not required_facts and not optional_facts,
        "note": note,
    }


def load_smoke_cases() -> list[dict]:
    """Four cheap development checks before running the frozen main set."""
    cases = [
        _case(
            "smoke_transactional_none",
            "Thanks, that answers my question. Let's continue.",
            required_facts=[],
            forbidden_claims=[
                "The user has a lasting preference for this topic or answer style."
            ],
            note="A routine acknowledgment should yield no durable memory.",
        ),
        _case(
            "smoke_default_editor",
            "My default code editor is Zed.",
            required_facts=[
                _fact(
                    "The user's default code editor is Zed.",
                    kind="evidence",
                    scope="user",
                )
            ],
            forbidden_claims=["The user's editor is VS Code."],
            note="One explicit durable tool fact with a short literal source.",
        ),
        _case(
            "smoke_corrected_tool_and_schedule",
            "I no longer use Trello; I now use Linear for project tracking. "
            "My weekly planning block is Tuesday morning.",
            required_facts=[
                _fact(
                    "The user now uses Linear for project tracking.",
                    kind="evidence",
                    scope="user",
                ),
                _fact(
                    "The user's weekly planning block is Tuesday morning.",
                    kind="evidence",
                    scope="user",
                ),
            ],
            forbidden_claims=[
                "The user currently uses Trello for project tracking.",
                "The planning block is a one-time Tuesday meeting.",
            ],
            optional_facts=[
                _fact("The user used Trello previously, as historical context.")
            ],
            note="Two independent durable facts; the old tool must not be treated as current.",
        ),
        _case(
            "smoke_chat_only_branch_and_style",
            "For this debugging chat only, the current branch is `patch/cedar-17`. "
            "In this chat, explain fixes as short numbered steps before detailed reasoning.",
            required_facts=[
                _fact(
                    "The current branch for this debugging chat is exactly `patch/cedar-17`.",
                    kind="evidence",
                    scope="chat",
                    preserve_verbatim=True,
                ),
                _fact(
                    "In this chat, the user wants fixes explained as short numbered steps before detailed reasoning.",
                    kind="contract_signal",
                    scope="chat",
                ),
            ],
            forbidden_claims=[
                "The user's global default branch is `patch/cedar-17`.",
                "The user always prefers numbered explanations across all chats.",
            ],
            note="Two distinct chat-only memories test evidence versus contract signal and prevent global promotion.",
        ),
    ]
    _validate(cases, expected_count=4)
    return cases


def load_cases() -> list[dict]:
    """Eight new, varied development cases for manual comparison of extractors."""
    handoff_sections = [
        "I still review release health every Wednesday morning before our weekly planning call. At today's meeting I opened with the error trend from the sample batch and asked the team to distinguish confirmed failures from dashboard noise. We agreed that the review rhythm still gives us enough time to correct issues before the next planning discussion.",
        "The operations lead walked through the agenda, starting with access, then reporting, then the migration decision. Several participants had arrived with separate notes, so we spent a few minutes aligning the names used for the test tenant and the sample batch. Nobody proposed a new production date at this stage of the meeting.",
        "The note taker asked whether the agenda should group all open checks under a single heading. We decided to keep access and data verification separate in the working document because they have different reviewers. The headings are only for today's review and do not introduce a new system status or a new approval category.",
        "A support colleague described two customer questions about historical records. We read the tickets aloud to decide whether they concerned export format or missing content, and postponed an answer until the data team could inspect the examples. The discussion produced a small follow-up task, not a change to the public product contract.",
        "Someone suggested placing the ticket examples directly in the presentation deck. The support lead preferred a short description until the underlying records were checked, since a screenshot could make the provisional interpretation look final. The group moved on without selecting a slide or changing the customer-facing explanation.",
        "The data team showed a count comparison from its rehearsal. One report grouped revisions by creation date and another used processing date, which explained most of the apparent discrepancy. We asked for the underlying rows to be kept in the working folder until the reconciliation is signed, rather than quoting a provisional total in a release note.",
        "The analyst demonstrated how the two grouping methods behave on a small fabricated ledger. That example helped the room understand the discrepancy but did not establish a production total. We asked for the chart to be labeled as a rehearsal illustration before using it in the internal meeting follow-up.",
        "During the documentation review, we found that the draft changelog included raw stack details that would distract readers looking for user-visible changes. I want the public changelog kept concise, but full diagnostic details belong in the internal incident note. The documentation owner accepted that split and will prepare another draft for review.",
        "We then checked the screenshot list for the help article. The button labels in the training image matched the staging interface, but one caption described the older account screen. We decided to replace that caption before anyone circulates the article, because the article should guide operators through the interface they will actually see.",
        "A teammate summarized a dry-run of the export path. The files opened correctly on the test machine, although the timestamp column still needs a clearer description for readers in different time zones. The group agreed to record the timestamp example in the issue tracker and revisit wording after the next sample export.",
        "The export discussion briefly turned to file naming. One participant proposed adding the run date to every filename, while another showed how that would affect scripts that compare successive samples. We left the naming proposal open and limited today's decision to documenting what the current sample already produces.",
        "The engineering owner reported that a worker recovered from an interrupted connection without losing the sample queue. We reviewed the trace briefly and agreed that the trace alone could not prove the final production procedure. The relevant question for today's handoff was whether the rehearsal could continue while its open checks were listed clearly.",
        "At the access section, Nia explained that the permission review covers both the service account and the operator role. She had checked the first set of grants, but the second set still required a comparison with the signed access list. We did not treat a partial check as approval, and the minutes identify the remaining comparison as her open task.",
        "The release coordinator asked how we would describe the current network route to people preparing the next rehearsal. We checked the service registry and the staging deployment record rather than relying on an old diagram. The answer was added to the handoff notes so the team will not accidentally point the sample client at a retired route.",
        "The current staging endpoint is `/v3/ledger?mode=dry-run`; the old `/v2/ledger` path is retired. The exact slash, question mark, and mode value matter to the sample client. We copied the route into the staging instructions, and we left the production endpoint out of that example because it was irrelevant to the rehearsal.",
        "After the route check, we discussed logging volume. One engineer suggested raising the debug level for a short test, while another warned that it might obscure the smaller warnings we needed to inspect. We did not settle that experiment, and the action was only to collect a representative baseline before anyone changes the logging setting.",
        "We also compared the ordinary dashboard view with a more detailed diagnostic panel. The detailed view would help a reviewer investigate a failed sample, but it might overwhelm operators who only need a go-or-stop status. The designer took both sketches away for another pass; the meeting did not select a permanent layout.",
        "The user interface review centered on wording for the dashboard. A designer showed two versions of a status sentence, and the team compared whether either one implied that an incomplete run was finished. We kept the simpler wording as a draft but asked for a fresh screenshot after the next build, since the current mockup was not final.",
        "The storage group described an object checksum check from the sample batch. The checksum comparison passed for the inspected files, but the group had not yet sampled older archived attachments. We wrote down that gap so the final verification report will not silently imply that every historical object was examined in the rehearsal.",
        "A records specialist asked whether the archive sample should include files with uncommon extensions. The group agreed to assemble a representative synthetic list before the next rehearsal, because the current set was chosen for convenience. This was a preparation task, not a claim that those uncommon files are presently missing or damaged.",
        "We reviewed the ownership table because two names appeared beside the same handoff item. I own the dashboard copy, while Nia owns the permission review. The copy can be prepared while access checks continue, but a polished dashboard sentence cannot substitute for her approval of the grants used by the full migration.",
        "A finance observer asked whether the sample replay would change any invoice totals. The team explained the boundaries of the synthetic batch and agreed to compare its output with the frozen sample ledger. We left the real settlement workflow untouched and recorded the observer's question for the later operational review.",
        "Another discussion concerned the rollback checklist. The draft had the right sequence but omitted where operators should record the reason for a stop. We asked its owner to add that line, then read the checklist from the perspective of a person who has not attended these meetings. No rollback was performed during today's review.",
        "The support lead noted that an early handoff email had called the sample rehearsal a migration. We corrected the wording in the meeting notes because the email could be read as describing a full rollout. For now the sample batch is evidence about the procedure, and the team still needs the separate permission decision and final scheduling step.",
        "The communications owner read the revised handoff sentence aloud to see whether it could be mistaken for an announcement. One version sounded too definitive, so the group kept a neutral working phrase and deferred external wording. We made no decision to notify customers or publish a release update from this meeting.",
        "Last night's failed sample job was retried successfully this morning. The replay reached the expected sample row count, but it did not exercise the complete production dataset. We agreed not to describe the whole migration as complete, even though the successful retry makes the next rehearsal easier to plan.",
        "A testing colleague described a filename edge case that appears only when a document title contains a percent sign. The group asked for a narrow reproduction using synthetic files and deferred any broader conclusion until the result is available. This item remains a working test question rather than a claim that every export path has failed.",
        "The privacy review covered which sample attachments could be shown in a training screenshot. We chose fabricated records for the illustration and asked the documentation owner to remove a draft image that had been copied from an internal test. The meeting did not authorize publishing any private source material or changing the production retention rule.",
        "At the end of the review, a teammate asked whether the training material needed a step-by-step video. The group considered the support cost and the chance that the interface might change before release. We decided only to gather examples of frequent operator questions, leaving the format of the eventual training material undecided.",
        "Before closing, the coordinator read back the open actions: finish the access comparison, revise the dashboard wording, clarify the timestamp example, and complete the checklist note. We checked that each action had a named owner in the working minutes. Several topics were still under discussion, so the readback was not presented as a final deployment approval.",
        "Nia has not signed off on the permission review, so the full migration must wait. After she signs off, we can schedule the full run, but there is no date for it yet. I asked the coordinator to keep that condition with the handoff summary so a reader will not mistake the successful sample retry for authorization to run the complete migration.",
    ]
    handoff_source = "\n\n".join(
        "\n".join(handoff_sections[index : index + 8])
        for index in range(0, len(handoff_sections), 8)
    )
    cases = [
        _case(
            "reply_language_with_exception",
            "Spanish, but keep code and API names in English. The rest can be translated.",
            recent_messages=[
                {
                    "role": "assistant",
                    "content": "For future coding help, which language should I use in my replies?",
                }
            ],
            required_facts=[
                _fact(
                    "For future coding help, the user wants replies in Spanish.",
                    kind="contract_signal",
                    scope="user",
                ),
                _fact(
                    "The user wants code and API names kept in English.",
                    kind="contract_signal",
                    scope="user",
                ),
            ],
            forbidden_claims=[
                "The user wants code identifiers translated into Spanish."
            ],
            note="An elliptical answer inherits its durable context from the preceding question while preserving an exception.",
        ),
        _case(
            "spanish_city_and_commute",
            "Desde marzo vivo en Zaragoza, no en Granada. Los martes y jueves voy al trabajo en tranvía; los demás días suelo caminar.",
            required_facts=[
                _fact(
                    "The user currently lives in Zaragoza rather than Granada.",
                    kind="evidence",
                    scope="user",
                ),
                _fact(
                    "The user commutes by tram on Tuesdays and Thursdays and usually walks on other days.",
                    kind="evidence",
                    scope="user",
                ),
            ],
            forbidden_claims=[
                "The user currently lives in Granada.",
                "The user takes the tram to work every day.",
            ],
            optional_facts=[
                _fact(
                    "The move to Zaragoza began in March, if kept with the current-city fact."
                )
            ],
            note="Spanish source with a corrected city and a qualified recurring commute pattern.",
        ),
        _case(
            "catalan_exact_membership_and_reminder",
            "Per a la biblioteca del barri, el meu codi de soci és CAT-17/4. Prefereixo que em recordis el termini de devolució dos dies abans.",
            required_facts=[
                _fact(
                    "The user's neighborhood library membership code is CAT-17/4.",
                    kind="evidence",
                    scope="user",
                    preserve_verbatim=True,
                ),
                _fact(
                    "The user prefers return-deadline reminders two days in advance.",
                    kind="contract_signal",
                    scope="user",
                ),
            ],
            forbidden_claims=["The membership code is CAT-17-4 or CAT-17/7."],
            note="Catalan durable facts; the code's slash, hyphen, and digits are material.",
        ),
        _case(
            "assistant_advice_stays_chat_scoped",
            "I recommended keeping the Azul deployment paused until the audit closes. The rollback log shows that the last attempt failed because the service account lacked write permission.",
            role="assistant",
            recent_messages=[
                {
                    "role": "user",
                    "content": "What should we do about the Azul deployment while the audit is open?",
                }
            ],
            required_facts=[
                _fact(
                    "The assistant advised keeping the Azul deployment paused until the audit closes.",
                    kind="evidence",
                    scope="chat",
                ),
                _fact(
                    "The assistant found that the last deployment attempt failed because the service account lacked write permission.",
                    kind="evidence",
                    scope="chat",
                ),
            ],
            forbidden_claims=[
                "The user personally prefers all deployments paused.",
                "The Azul deployment has already resumed.",
            ],
            note="Assistant-authored decision and root-cause finding are chat evidence, not a global user trait or completed action.",
        ),
        _case(
            "hypothetical_translation_none",
            "Could you translate the sample sentence 'I prefer early flights' into French? It is only a translation exercise.",
            required_facts=[],
            forbidden_claims=[
                "The user prefers early flights.",
                "The user prefers to communicate in French.",
            ],
            note="Quoted exercise text and a one-off translation request should not create personal memories.",
        ),
        _case(
            "archive_project_with_temporary_interruption",
            "I coordinate the Atlas photo archive for our museum, and I review the catalog every Monday morning. "
            "The public labels use the collection title 'City After Rain'; please keep that spelling when we discuss the exhibit. "
            "Today the office Wi-Fi is down, so I may send files later than usual. The outage began this morning and the technician is already checking it. "
            "For the archive itself, we store consent forms separately from image captions because those records have different reviewers. "
            "Next week I plan to test a new scanner, but I have not chosen a model or bought anything yet.",
            required_facts=[
                _fact(
                    "The user coordinates the museum's Atlas photo archive.",
                    kind="evidence",
                    scope="user",
                ),
                _fact(
                    "The user reviews that catalog every Monday morning.",
                    kind="evidence",
                    scope="user",
                ),
                _fact(
                    "The exhibit's collection title is exactly 'City After Rain'.",
                    kind="evidence",
                    scope="user",
                    preserve_verbatim=True,
                ),
                _fact(
                    "The archive stores consent forms separately from image captions because they have different reviewers.",
                    kind="evidence",
                    scope="user",
                ),
            ],
            forbidden_claims=[
                "The office Wi-Fi is permanently unavailable.",
                "The user has purchased a new scanner.",
            ],
            optional_facts=[
                _fact(
                    "The office Wi-Fi is down today, if represented only as a temporary state.",
                    kind="state_update",
                    temporal_type="ephemeral",
                ),
                _fact(
                    "The user plans to test a scanner next week, without claiming a model has been selected."
                ),
            ],
            note="A multitheme update contrasts recurring archive facts with a temporary outage and an uncommitted purchase idea.",
        ),
        _case(
            "spanish_event_with_third_party_correction",
            "Para el ciclo de charlas, yo llevo la coordinación de ponentes y Sara se ocupa del sonido. "
            "El primer borrador decía que el ensayo sería el viernes a las 18:00, pero lo cambiamos al sábado a las 10:30 en la sala B. "
            "Mi compañero dijo que todos los invitados habían confirmado; en realidad faltan dos respuestas, así que no anuncies la lista como cerrada. "
            "Cada mes prefiero revisar el programa en una tabla breve antes de preparar el cartel. "
            "La clave interna del evento es EVT-24/09; mantenla tal cual en el documento de producción.",
            required_facts=[
                _fact(
                    "The user coordinates speakers for the talk series, while Sara handles sound.",
                    kind="evidence",
                    scope="user",
                ),
                _fact(
                    "The rehearsal was moved from Friday to Saturday at 10:30 in room B.",
                    kind="evidence",
                    scope="user",
                ),
                _fact(
                    "Two guest confirmations are still missing, so the guest list is not final.",
                    kind="evidence",
                    scope="user",
                ),
                _fact(
                    "The user prefers a brief table for the monthly program review before making the poster.",
                    kind="contract_signal",
                    scope="user",
                ),
                _fact(
                    "The event's internal key is EVT-24/09.",
                    kind="evidence",
                    scope="user",
                    preserve_verbatim=True,
                ),
            ],
            forbidden_claims=[
                "The rehearsal remains on Friday at 18:00.",
                "All guests have confirmed.",
                "The user is responsible for sound instead of speakers.",
            ],
            note="Spanish multitheme planning distinguishes current schedule, speaker attribution, correction, preference, and literal key.",
        ),
        _case(
            "multi_chunk_conditional_handoff",
            handoff_source,
            recent_messages=[
                {
                    "role": "assistant",
                    "content": "Does your Wednesday release review still fit, and what changed in the handoff?",
                }
            ],
            required_facts=[
                _fact(
                    "The user continues the Wednesday morning release-health review rhythm.",
                    kind="evidence",
                    scope="user",
                ),
                _fact(
                    "The user wants a concise public changelog and full diagnostics in the internal incident note.",
                    kind="contract_signal",
                    scope="user",
                ),
                _fact(
                    "The staging endpoint is exactly `/v3/ledger?mode=dry-run`; `/v2/ledger` is retired.",
                    kind="evidence",
                    scope="user",
                    preserve_verbatim=True,
                ),
                _fact(
                    "The user owns dashboard copy and Nia owns permission review.",
                    kind="evidence",
                    scope="user",
                ),
                _fact(
                    "The full migration waits for Nia's sign-off and has no scheduled date.",
                    kind="evidence",
                    scope="user",
                ),
            ],
            forbidden_claims=[
                "Nia has already approved the permission review.",
                "The full migration completed this morning.",
                "The retired `/v2/ledger` path is still current.",
            ],
            optional_facts=[
                _fact(
                    "The failed sample job succeeded on retry this morning, without implying the full run completed."
                )
            ],
            note="Authentic-style meeting notes distribute five durable facts across chunks; ordinary discussion may be omitted, and the full migration remains conditional.",
        ),
    ]
    _validate(cases, expected_count=8)
    if sum(not case["required_facts"] for case in cases) < 1:
        raise ValueError("The main set needs at least one no-memory case")
    return cases


def _validate(cases: list[dict], *, expected_count: int) -> None:
    if (
        len(cases) != expected_count
        or len({case["case_id"] for case in cases}) != expected_count
    ):
        raise ValueError("Incorrect or duplicate extractor case IDs")
    for case in cases:
        if case["role"] not in {"user", "assistant"}:
            raise ValueError(f"Invalid source role: {case['case_id']}")
        if len(case["required_facts"]) > 6:
            raise ValueError(f"Too many facts for manual review: {case['case_id']}")
        if case["expected_nothing_durable"] and case["required_facts"]:
            raise ValueError(f"Inconsistent no-memory label: {case['case_id']}")
    json.dumps(cases, ensure_ascii=False)
