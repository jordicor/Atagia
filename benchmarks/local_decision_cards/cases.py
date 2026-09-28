"""Frozen synthetic development cases for three narrow production decisions.

These labels are author-defined expectations, not independent human gold. The
context cases exercise ``decide_reuse`` for local/native-choice challengers.
The OpenRouter Luna baseline is limited to the production generative exact and
sentiment cards because OpenRouter does not support ``choice_questions``.
Production's older generative staleness-signal route is a different task.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings, default_resource_path
from atagia.memory.consequence_detector import (
    ConsequenceCardResult,
    ConsequenceDetector,
    _sentiment_decision,
)
from atagia.memory.context_staleness import ContextStalenessRequest, ContextStalenessSignalDetector
from atagia.memory.need_choices import decode_need_choices
from atagia.memory.need_detector import (
    NeedDetector,
    _authority_context_from_extraction_context,
    _parse_card_output,
)
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver, compute_effective_policy_hash
from atagia.models.schemas_cache import ContextCacheEntry
from atagia.models.schemas_decisions import ChoiceAnswer
from atagia.models.schemas_memory import (
    ConsequenceSentiment,
    ExtractionContextMessage,
    ExtractionConversationContext,
)
from atagia.services.llm_client import LLMCompletionRequest, LLMCompletionResponse
from atagia.services.prompt_authority import process_authority_context

Card = Literal["context_reuse", "need_detector_exact", "consequence_sentiment"]
Split = Literal["smoke", "evaluation", "robustness"]
Language = Literal["en", "es", "ca"]


@dataclass(frozen=True, slots=True)
class DecisionCase:
    case_id: str
    split: Split
    card: Card
    language: Language
    previous_message: str
    message: str
    expected: str
    rationale: str


# Each tuple is (id, language, previous message, current message, expected, rationale).
# The cases are fixed before model evaluation and must not be tuned to its outputs.
_SMOKE: dict[Card, tuple[tuple[str, Language, str, str, str, str], ...]] = {
    "context_reuse": (
        ("context-smoke-1", "en", "Let's fix the login test failure.", "Continue with that login test.", "reuse", "The subject and perspective continue."),
        ("context-smoke-2", "es", "Revisemos el error de acceso de Ana.", "No, me refería al acceso de Luis.", "refresh", "The user corrects the person in the cached context."),
    ),
    "need_detector_exact": (
        ("exact-smoke-1", "en", "", "What date did I say the workshop starts?", "yes", "The answer requires a date supplied by the user."),
        ("exact-smoke-2", "es", "", "¿Cuántos días tiene una semana?", "no", "General knowledge suffices."),
    ),
    "consequence_sentiment": (
        ("sentiment-smoke-1", "en", "Try clearing the local cache.", "Clearing the cache fixed the error.", "positive", "The reported result succeeded."),
        ("sentiment-smoke-2", "es", "Prueba a reiniciar el proceso.", "Lo reinicié y ahora falla también el inicio de sesión.", "negative", "The reported result made things worse."),
    ),
}

_EVALUATION: dict[Card, tuple[tuple[str, Language, str, str, str, str], ...]] = {
    "context_reuse": (
        ("context-en-1", "en", "Let's adjust the invoice layout.", "Can you make the invoice heading larger?", "reuse", "Same invoice layout task."),
        ("context-en-2", "en", "We are reviewing the tablet battery issue.", "Keep going with the battery checks.", "reuse", "Direct continuation of the same issue."),
        ("context-en-3", "en", "Use the blue label for the archive box.", "Actually, the label must be green.", "refresh", "A prior detail is explicitly corrected."),
        ("context-en-4", "en", "Let's plan the garden irrigation.", "Switch to planning the exhibition budget.", "refresh", "The user changes subjects."),
        ("context-es-1", "es", "Estamos corrigiendo la tabla de pedidos.", "Sigue con la columna de fecha de esa tabla.", "reuse", "Continues the same table task."),
        ("context-es-2", "es", "Revisamos el ruido del ventilador.", "¿Y cómo comprobamos ese mismo ventilador?", "reuse", "Refers to the same device and issue."),
        ("context-es-3", "es", "El evento será en Valencia.", "Perdón, será en Sevilla.", "refresh", "The event location is corrected."),
        ("context-es-4", "es", "Hablemos del menú de la fiesta.", "Ahora quiero revisar el contrato de alquiler.", "refresh", "A new subject requires fresh context."),
        ("context-ca-1", "ca", "Estem arreglant la pantalla d'inici.", "Continua amb el botó d'aquesta pantalla.", "reuse", "Continues the same interface task."),
        ("context-ca-2", "ca", "Mirem el retard del tren de matí.", "Què més sabem d'aquest mateix tren?", "reuse", "Keeps the same train as referent."),
        ("context-ca-3", "ca", "La reunió és dimarts.", "No, m'he equivocat: és dijous.", "refresh", "The previous date is negated."),
        ("context-ca-4", "ca", "Parlem del pressupost del jardí.", "Canviem de tema: vull preparar una entrevista.", "refresh", "The topic changes explicitly."),
    ),
    "need_detector_exact": (
        ("exact-en-1", "en", "I named the new studio Hazel Room.", "What name did I choose for the studio?", "yes", "Requires the user's chosen name."),
        ("exact-en-2", "en", "The meeting is on the twelfth.", "Which day did I just give for the meeting?", "yes", "Requires a detail from this chat."),
        ("exact-en-3", "en", "", "What is the capital of Portugal?", "no", "Public knowledge suffices."),
        ("exact-en-4", "en", "I mentioned my bicycle earlier.", "How does a bicycle gear work in general?", "no", "The question asks for a general explanation, not a remembered bicycle detail."),
        ("exact-es-1", "es", "Elegí el código M42 para la caja.", "¿Qué código elegí para la caja?", "yes", "Requires the user's code."),
        ("exact-es-2", "es", "", "Repite las palabras exactas que usé para el título.", "yes", "Requires the user's prior wording."),
        ("exact-es-3", "es", "", "¿Qué es una célula vegetal?", "no", "A general definition suffices."),
        ("exact-es-4", "es", "Comenté mi receta ayer.", "¿Qué es una receta de cocina?", "no", "The user's earlier recipe is not needed for a general definition."),
        ("exact-ca-1", "ca", "Vaig dir que el paquet arriba el dia 18.", "Quin dia vaig dir que arribava el paquet?", "yes", "Requires the user's stated date."),
        ("exact-ca-2", "ca", "", "Quina contrasenya et vaig donar per al wifi?", "yes", "Requires a previously supplied password."),
        ("exact-ca-3", "ca", "", "Quants costats té un triangle?", "no", "Public knowledge suffices."),
        ("exact-ca-4", "ca", "Et vaig parlar del meu telescopi.", "Com funciona un telescopi en general?", "no", "General mechanics do not need the user's saved detail."),
    ),
    "consequence_sentiment": (
        ("sentiment-en-1", "en", "Try the smaller image file.", "The smaller image loads instantly now.", "positive", "The reported result succeeded."),
        ("sentiment-en-2", "en", "Rename the column before importing.", "Renaming the column solved the import error.", "positive", "The import error was resolved."),
        ("sentiment-en-3", "en", "Restart the print service.", "After the restart, printing stopped entirely.", "negative", "The result is worse."),
        ("sentiment-en-4", "en", "Change the chart scale.", "The labels are readable now, but half the points disappeared.", "neutral", "One part improved and another failed."),
        ("sentiment-es-1", "es", "Prueba la versión nueva del controlador.", "Con el controlador nuevo ya funciona el escáner.", "positive", "The scanner now works."),
        ("sentiment-es-2", "es", "Reduce el tamaño del archivo.", "Al reducirlo, se envió sin errores.", "positive", "The file sent successfully."),
        ("sentiment-es-3", "es", "Cambia la ruta de salida.", "La cambié y se perdieron los archivos generados.", "negative", "The outcome caused a loss."),
        ("sentiment-es-4", "es", "Ajusta el brillo de la pantalla.", "Ahora se ve mejor de día, pero por la noche deslumbra.", "neutral", "The result has both good and bad effects."),
        ("sentiment-ca-1", "ca", "Actualitza el connector.", "Amb el connector actualitzat, la sincronització funciona.", "positive", "Synchronization now works."),
        ("sentiment-ca-2", "ca", "Mou el cable a l'altre port.", "L'he canviat de port i ja detecta el disc.", "positive", "The disk is detected."),
        ("sentiment-ca-3", "ca", "Prova de reiniciar la càmera.", "Després de reiniciar-la, ja no grava res.", "negative", "The camera no longer records."),
        ("sentiment-ca-4", "ca", "Canvia el format del document.", "Ara s'obre bé, però les imatges surten borroses.", "neutral", "The result combines improvement and degradation."),
    ),
}


# Declared and frozen on 2026-09-25 UTC, before any model output was inspected.
# Long cases use repeated neutral page descriptions as deliberate input length,
# with the only task-decisive statement at the end. No runtime semantic rule is
# derived from the padding text.
ROBUSTNESS_CASES_FROZEN_ON_UTC = "2026-09-25"
_SPANISH_PAGE_PADDING = (
    "Esta página de papel cuadriculado tiene márgenes iguales, líneas paralelas, "
    "números de página consecutivos y espacios en blanco entre grupos de trazos. "
)
_CATALAN_PAGE_PADDING = (
    "Aquesta pàgina de paper quadriculat té marges iguals, línies paral·leles, "
    "números de pàgina consecutius i espais en blanc entre grups de traços. "
)


def _long_message(padding: str, final_statement: str) -> str:
    return padding * 64 + final_statement


_ROBUSTNESS: dict[Card, tuple[tuple[str, Language, str, str, str, str], ...]] = {
    "context_reuse": (
        ("context-robust-es-1", "es", "El informe de Nora usa cifras de abril.", "Sobre el informe de Nora, no uses cifras de abril: eran las de marzo.", "refresh", "The same report and person remain, but the month is corrected."),
        ("context-robust-ca-2", "ca", "Ara acabem la llista de mobiliari de l'oficina.", "No seguim amb el mobiliari: torna a la proposta de contracte que vam deixar ahir.", "refresh", "An explicit older reference replaces the current subject."),
        ("context-robust-en-3", "en", "Let's debug the new export routine.", "The code sample contains the quoted string 'do not export'; that is test data. Please keep debugging the same export routine.", "reuse", "The quoted negation is data; the user continues the same task."),
        ("context-robust-es-long", "es", "Estamos revisando las medidas del cartel para la exposición de otoño.", _long_message(_SPANISH_PAGE_PADDING, "Corrección final: no quiero seguir con ese cartel de otoño; vuelve al plan anterior de transporte para la feria de primavera."), "refresh", "The final instruction explicitly switches from the cached poster task to an older transport plan."),
    ),
    "need_detector_exact": (
        ("exact-robust-es-1", "es", "Mi huerto usa el riego por goteo modelo R7.", "Aunque mi huerto tenga el modelo R7, ¿cómo funciona el riego por goteo en general?", "no", "The personal model is cited but the answer asks for general mechanics."),
        ("exact-robust-ca-2", "ca", "Vaig apuntar la taula 14 per a la reserva.", "No vull una explicació general: quin número de taula vaig apuntar per a la reserva?", "yes", "The answer requires the table number supplied by the user."),
        ("exact-robust-en-3", "en", "I said my bicycle has 37 mm tires.", "The note 'my bicycle has 37 mm tires' is context; what does tire pressure do in general?", "no", "The quoted personal specification is unnecessary for the general answer."),
        ("exact-robust-ca-long", "ca", "Vaig triar el nom Cingle Blau per a la sala de lectura.", _long_message(_CATALAN_PAGE_PADDING, "La pregunta final és: quin nom exacte vaig triar per a la sala de lectura?"), "yes", "The final question requires the user's previously chosen name."),
    ),
    "consequence_sentiment": (
        ("sentiment-robust-es-1", "es", "Prueba a reducir el tamaño del PDF.", "El registro cita literalmente 'no funcionó' de una prueba anterior; con el PDF reducido ahora se envía correctamente.", "positive", "The negative quotation describes an earlier test; the reported current result succeeded."),
        ("sentiment-robust-ca-2", "ca", "Prova de canviar la carpeta de destinació.", "Al fitxer hi ha la frase 'ha fallat' com a exemple; després del canvi de carpeta ara es desa bé.", "positive", "The negative phrase is example text; the current save succeeds."),
        ("sentiment-robust-es-3", "es", "Cambia el orden de las columnas.", "El mensaje de prueba incluía la palabra 'éxito', pero al aplicar tu cambio se perdieron las columnas de fecha.", "negative", "The positive word is quoted test data; the actual outcome lost columns."),
        ("sentiment-robust-ca-long", "ca", "Canvia el format d'exportació a CSV.", _long_message(_CATALAN_PAGE_PADDING, "Resultat final: després de canviar a CSV, l'exportació ha fallat del tot i no es pot obrir cap fitxer."), "negative", "The final reported outcome is a complete export failure."),
    ),
}


def _materialize(split: Split, source: dict[Card, tuple[tuple[str, Language, str, str, str, str], ...]]) -> tuple[DecisionCase, ...]:
    return tuple(
        DecisionCase(case_id, split, card, language, previous, message, expected, rationale)
        for card, rows in source.items()
        for case_id, language, previous, message, expected, rationale in rows
    )


SMOKE_CASES = _materialize("smoke", _SMOKE)
EVALUATION_CASES = _materialize("evaluation", _EVALUATION)
ROBUSTNESS_CASES = _materialize("robustness", _ROBUSTNESS)


class _CapturingClient:
    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        if request.choice_questions:
            answers = {
                key: ChoiceAnswer(
                    type="choice",
                    choice=next(iter(question.criteria)),
                    probabilities={choice: float(choice == next(iter(question.criteria))) for choice in question.criteria},
                    confidence=1.0,
                )
                for key, question in request.choice_questions.items()
            }
            return LLMCompletionResponse(provider="fixture", model=request.model, choice_answers=answers)
        return LLMCompletionResponse(provider="fixture", model=request.model, output_text="no")


def _context(case: DecisionCase) -> ExtractionConversationContext:
    return ExtractionConversationContext(
        user_id="local_decision_user",
        conversation_id=f"local_decision_{case.case_id}",
        source_message_id=f"source_{case.case_id}",
        assistant_mode_id="general_qa",
        recent_messages=(
            [ExtractionContextMessage(
                role="assistant" if case.card == "consequence_sentiment" else "user",
                content=case.previous_message,
            )]
            if case.previous_message else []
        ),
        privacy_enforcement="off",
    )


async def capture_request(case: DecisionCase, model_spec: str) -> LLMCompletionRequest:
    """Capture exactly one production card request without any provider call.

    ``model_spec`` is the route under test. The context card always uses its
    finite-choice ``decide_reuse`` builder; OpenRouter is ineligible there.
    """
    if case.card == "context_reuse" and model_spec.startswith("openrouter/"):
        raise ValueError("OpenRouter does not support context_reuse choice_questions")
    client = _CapturingClient()
    settings = replace(
        Settings.from_env({}),
        llm_forced_global_model=None,
        llm_finite_decisions_enabled=True,
        llm_component_models={
            "context_staleness": model_spec,
            "need_detector_exact": model_spec,
            "consequence_sentiment": model_spec,
        },
    )
    manifests = ManifestLoader(Path(default_resource_path("manifests"))).load_all()
    policy = PolicyResolver().resolve(manifests["general_qa"], None, None)
    if case.card == "context_reuse":
        profile = {
            "profile_id": "normal", "signals": {}, "risk_level": "normal",
            "authorized": True, "profile_hash": "synthetic", "token": "synthetic",
        }
        entry = ContextCacheEntry.model_validate({
            "cache_key": "ctx:synthetic", "user_id": "local_decision_user",
            "lifecycle_epoch": "synthetic", "cache_revision": 1, "derivation_revision": 1,
            "conversation_id": f"local_decision_{case.case_id}",
            "assistant_mode_id": "general_qa", "policy_prompt_hash": policy.prompt_hash,
            "effective_policy_hash": compute_effective_policy_hash(policy),
            "operational_profile": profile,
            "composed_context": {
                "contract_block": "", "workspace_block": "", "memory_block": "",
                "state_block": "", "selected_memory_ids": [],
                "total_tokens_estimate": 0, "budget_tokens": 500,
                "items_included": 0, "items_dropped": 0,
            },
            "cached_at": "2026-09-24T12:00:00+00:00", "last_retrieval_message_seq": 1,
            "last_user_message_text": case.previous_message,
        })
        request = ContextStalenessRequest(
            user_id="local_decision_user", conversation_id=f"local_decision_{case.case_id}",
            message_text=case.message, current_message_seq=2,
            operational_profile=profile, effective_policy_hash=compute_effective_policy_hash(policy),
        )
        await ContextStalenessSignalDetector(client, settings).decide_reuse(
            cache_entry=entry, request=request, resolved_policy=policy,
        )
    elif case.card == "need_detector_exact":
        context = _context(case)
        detector = NeedDetector(client, FrozenClock(datetime(2026, 9, 24, tzinfo=timezone.utc)), settings)
        await detector._run_card(
            card_name="exact", message_text=case.message, role="user", context=context,
            resolved_policy=policy, content_language_profile=[], user_communication_profile=None,
            prompt_authority_context=_authority_context_from_extraction_context(
                context, purpose="need_detection",
            ),
        )
    elif case.card == "consequence_sentiment":
        context = _context(case)
        detector = ConsequenceDetector(
            client, FrozenClock(datetime(2026, 9, 24, tzinfo=timezone.utc)), settings,
        )
        authority = process_authority_context(
            privacy_enforcement=context.privacy_enforcement, user_id=context.user_id,
            privilege_level=context.authenticated_user_privilege_level,
            is_atagia_master=context.authenticated_user_is_atagia_master,
            purpose="consequence_detection",
        )
        await detector._run_card(
            card_name="sentiment", message_text=case.message, role="user",
            conversation_context=context,
            recent_assistant_messages=[{"id": "assistant_1", "text": case.previous_message}],
            authority_context=authority,
        )
    else:
        raise ValueError(f"Unsupported card: {case.card}")
    if len(client.requests) != 1:
        raise AssertionError(f"Expected one captured request, got {len(client.requests)}")
    return client.requests[0]


def score_response(
    case: DecisionCase,
    request: LLMCompletionRequest,
    response: LLMCompletionResponse,
) -> tuple[str | None, bool, bool]:
    """Return (decision, parse_valid, matches_expected) for one raw card answer.

    Native choices use the production decoders. Generative exact/sentiment
    answers use production parsers. A plain-text context label is accepted as a
    harness adapter only when it exactly names one finite-choice option.
    """
    if case.card == "context_reuse":
        question = (request.choice_questions or {}).get("context_reuse")
        if question is None:
            raise ValueError("Context reuse requires the production finite-choice request")
        if response.choice_answers:
            if set(response.choice_answers) != {"context_reuse"}:
                return None, False, False
            decision = response.choice_answers["context_reuse"].choice
        else:
            decision = response.output_text.strip().lower()
        valid = decision in question.criteria
        return (decision if valid else None), valid, valid and decision == case.expected
    if case.card == "need_detector_exact":
        if request.choice_questions is not None:
            try:
                decision = decode_need_choices("exact", response.choice_answers, request.choice_questions)["exact_recall_needed"]
            except (KeyError, ValueError, RuntimeError):
                return None, False, False
            label = "yes" if decision else "no"
            return label, True, label == case.expected
        parsed, valid = _parse_card_output("exact", response.output_text)
        decision = parsed.get("exact_recall_needed")
        label = ("yes" if decision else "no") if valid else None
        return label, valid, valid and label == case.expected
    if case.card == "consequence_sentiment":
        if request.choice_questions is not None:
            if set(response.choice_answers) != {"outcome_sentiment"}:
                return None, False, False
            choice = response.choice_answers["outcome_sentiment"].choice
            if choice not in request.choice_questions["outcome_sentiment"].criteria:
                return None, False, False
            result = ConsequenceCardResult("sentiment", choice=choice)
        else:
            result = ConsequenceCardResult("sentiment", raw_output=response.output_text)
        sentiment = _sentiment_decision(result)
        valid = sentiment != ConsequenceSentiment.NEUTRAL or (
            result.choice == ConsequenceSentiment.NEUTRAL.value
            or response.output_text.strip().lower().split(maxsplit=1)[0:1] == ["mixed"]
        )
        label = sentiment.value if valid else None
        return label, valid, valid and label == case.expected
    raise ValueError(f"Unsupported card: {case.card}")
