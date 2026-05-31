"""Schema tests for user communication language profiles."""

from __future__ import annotations

import pytest

from atagia.models.schemas_memory import (
    ExplicitLanguageAbility,
    ExplicitLanguagePreference,
    LanguageContextualNorm,
    LanguageProfileSourceRef,
    ObservedUserLanguage,
    UserCommunicationProfile,
    UserCommunicationProfileTrace,
)


def _source_ref() -> LanguageProfileSourceRef:
    return LanguageProfileSourceRef(
        source_kind="source_message",
        source_message_id="msg_1",
        conversation_id="conv_1",
    )


def test_user_communication_profile_separates_observation_preference_and_ability() -> None:
    profile = UserCommunicationProfile(
        subject_presence_id=" human_owner ",
        observed_user_languages=[
            ObservedUserLanguage(
                language_code=" ES ",
                message_count=12,
                last_seen_at="2026-05-20T10:00:00+00:00",
                context_label=" ordinary_chat ",
                source_refs=[_source_ref()],
                confidence=0.91,
            )
        ],
        explicit_language_preferences=[
            ExplicitLanguagePreference(
                language_code="ca",
                preference_kind="contextual_answer_language",
                context_label="personal_chat",
                source_refs=[_source_ref()],
                confidence=0.84,
            )
        ],
        explicit_language_abilities=[
            ExplicitLanguageAbility(
                language_code="EN",
                ability_kind="understands",
                source_refs=[_source_ref()],
                confidence=0.9,
            )
        ],
        contextual_norms=[
            LanguageContextualNorm(
                language_code="en",
                norm_kind="comfortable_for_terms_or_code",
                context_label="work_technical",
                source_refs=[_source_ref()],
                confidence=0.8,
            )
        ],
    )

    assert profile.profile_kind == "user_language_profile"
    assert profile.subject_presence_id == "human_owner"
    assert profile.observed_user_languages[0].language_code == "es"
    assert profile.explicit_language_preferences[0].language_code == "ca"
    assert profile.explicit_language_abilities[0].language_code == "en"
    assert profile.contextual_norms[0].language_code == "en"
    assert profile.external_content_languages_excluded is True
    assert profile.control_plane_only is True


def test_user_communication_profile_requires_source_backing() -> None:
    with pytest.raises(Exception):
        ObservedUserLanguage(
            language_code="es",
            message_count=1,
            source_refs=[],
            confidence=0.8,
        )
    with pytest.raises(Exception):
        ExplicitLanguagePreference(
            language_code="es",
            preference_kind="default_answer_language",
            source_refs=[],
            confidence=0.8,
        )
    with pytest.raises(Exception):
        ExplicitLanguageAbility(
            language_code="es",
            ability_kind="speaks",
            source_refs=[],
            confidence=0.8,
        )


def test_user_communication_profile_rejects_external_content_as_observed_user_language() -> None:
    with pytest.raises(Exception):
        ObservedUserLanguage(
            language_code="zh",
            evidence_kind="external_artifact",
            message_count=1,
            source_refs=[_source_ref()],
            confidence=0.7,
        )


def test_user_communication_profile_rejects_unknown_or_malformed_language_codes() -> None:
    for bad_code in ("unknown", "eng", "1s", "", "jp", "zz"):
        with pytest.raises(Exception):
            ObservedUserLanguage(
                language_code=bad_code,
                message_count=1,
                source_refs=[_source_ref()],
                confidence=0.7,
            )


def test_user_communication_profile_control_plane_markers_are_literal_true() -> None:
    with pytest.raises(Exception):
        UserCommunicationProfile(control_plane_only=False)
    with pytest.raises(Exception):
        UserCommunicationProfile(external_content_languages_excluded=False)
    with pytest.raises(Exception):
        UserCommunicationProfile(stale=True)

    stale_profile = UserCommunicationProfile(
        stale=True,
        stale_reason="source_message_deleted",
    )
    assert stale_profile.stale is True
    assert stale_profile.stale_reason == "source_message_deleted"


def test_language_profile_source_refs_require_concrete_provenance() -> None:
    with pytest.raises(Exception):
        LanguageProfileSourceRef(source_kind="memory_object")
    with pytest.raises(Exception):
        LanguageProfileSourceRef(source_kind="source_message")
    with pytest.raises(Exception):
        LanguageProfileSourceRef(
            source_kind="message_window",
            conversation_id="conv_1",
            from_seq=5,
            to_seq=4,
        )

    ref = LanguageProfileSourceRef(
        source_kind="message_window",
        conversation_id="conv_1",
        from_seq=2,
        to_seq=4,
    )
    assert ref.conversation_id == "conv_1"


def test_user_communication_profile_trace_is_content_minimal() -> None:
    trace = UserCommunicationProfileTrace(
        profile_version=1,
        observed_language_codes=[" ES ", "es", "ca"],
        preference_language_codes=["ca"],
        ability_language_codes=["en"],
        contextual_norm_language_codes=["en"],
    )

    payload = trace.model_dump(mode="json")
    assert payload == {
        "profile_kind": "user_language_profile",
        "profile_version": 1,
        "stale": False,
        "observed_language_codes": ["es", "ca"],
        "preference_language_codes": ["ca"],
        "ability_language_codes": ["en"],
        "contextual_norm_language_codes": ["en"],
        "control_plane_only": True,
    }
    with pytest.raises(Exception):
        UserCommunicationProfileTrace(profile_version=1, control_plane_only=False)
