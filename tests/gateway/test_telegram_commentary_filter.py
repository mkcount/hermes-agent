from gateway.run import _telegram_interim_commentary_visible
from gateway.display_config import resolve_display_setting


def test_telegram_commentary_keeps_korean_narration():
    assert _telegram_interim_commentary_visible(
        "관련 구조를 확인하고 있습니다."
    )
    assert _telegram_interim_commentary_visible(
        "테스트를 실행합니다: pytest"
    )


def test_telegram_commentary_hides_english_status_summaries():
    assert not _telegram_interim_commentary_visible(
        "Planning direct image upload via Telegram API"
    )
    assert not _telegram_interim_commentary_visible(
        "Recommending direct upload via gateway adapter"
    )


def test_korean_commentary_filter_is_an_explicit_platform_override():
    config = {
        "display": {
            "platforms": {
                "telegram": {
                    "interim_commentary_language": "korean",
                }
            }
        }
    }
    assert (
        resolve_display_setting(
            config,
            "telegram",
            "interim_commentary_language",
        )
        == "korean"
    )
    assert (
        resolve_display_setting(
            {},
            "telegram",
            "interim_commentary_language",
        )
        == "all"
    )
