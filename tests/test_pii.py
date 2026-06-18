from app.observability.pii import prompt_hash, redact_pii


def test_redact_pii_masks_email_phone_and_card() -> None:
    raw = "Мой email ivan@mail.ru, тел +7 (999) 123-45-67, карта 4111 1111 1111 1111"

    masked = redact_pii(raw)

    assert "ivan@mail.ru" not in masked
    assert "+7 (999) 123-45-67" not in masked
    assert "4111 1111 1111 1111" not in masked

    assert "[EMAIL]" in masked
    assert "[PHONE_RU]" in masked
    assert "[CARD]" in masked


def test_redact_pii_masks_inn_and_passport() -> None:
    raw = "ИНН 7707083893, паспорт 4510 123456"

    masked = redact_pii(raw)

    assert "7707083893" not in masked
    assert "4510 123456" not in masked

    assert "[INN]" in masked
    assert "[PASSPORT]" in masked


def test_prompt_hash_is_stable_and_does_not_contain_raw_text() -> None:
    raw = "secret prompt with email user@example.com"

    first_hash = prompt_hash(raw)
    second_hash = prompt_hash(raw)

    assert first_hash == second_hash
    assert first_hash.startswith("sha256:")
    assert "user@example.com" not in first_hash