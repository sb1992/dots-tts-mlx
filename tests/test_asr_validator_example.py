"""Cover the pure-python core of ``examples/asr_validated_long.py``.

The example is not part of the installed package, so it is loaded by path. Nothing here
touches mlx-whisper or the model: the ASR import inside the example is lazy precisely so
the WER gate stays testable on its own.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "examples" / "asr_validated_long.py"

pytestmark = pytest.mark.skipif(not EXAMPLE.exists(), reason=f"example absent at {EXAMPLE}")


def _load():
    spec = importlib.util.spec_from_file_location("asr_validated_long", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_strips_case_and_punctuation():
    mod = _load()
    assert mod.normalize_words("Hello, world!  It's fine.") == ["hello", "world", "it", "s", "fine"]
    assert mod.normalize_words("") == []
    assert mod.normalize_words(None) == []


@pytest.mark.parametrize(
    "reference,hypothesis,expected",
    [
        ("one two three", "one two three", 0.0),          # exact
        ("one two three", "One, two THREE!", 0.0),        # normalization only
        ("one two three", "one two", 1 / 3),              # deletion
        ("one two three", "one two five", 1 / 3),         # substitution
        ("one two three", "one two three four", 1 / 3),   # insertion
        ("one two three", "", 1.0),                       # heard nothing
        ("", "", 0.0),                                    # both empty
        ("", "hallucinated words", 1.0),                  # pure insertion
    ],
)
def test_word_error_rate(reference, hypothesis, expected):
    mod = _load()
    assert mod.word_error_rate(reference, hypothesis) == pytest.approx(expected)


def test_wer_catches_same_length_hallucination():
    """The failure mode the hook exists for, and the reason this uses WER not coverage.

    A clip that says every intended word AND a sentence of invented text scores perfect
    word *coverage* — recall-only metrics cannot see insertions. WER can.
    """
    mod = _load()
    reference = "the meeting starts at noon"
    hallucinated = "the meeting starts at noon and then we all went to the beach"
    assert mod.word_error_rate(reference, hallucinated) > 0.5


def test_validator_accepts_and_rejects_against_the_threshold(monkeypatch):
    """Drive __call__ with a stubbed ASR — no mlx-whisper, no audio."""
    mod = _load()
    validator = mod.WhisperWerValidator("unused/repo", threshold=0.3, language="EN")
    assert validator.language == "en"  # whisper wants a lowercase code

    heard = {"text": "one two three"}
    monkeypatch.setattr(type(validator), "transcribe", lambda self, a, sr: heard["text"])
    audio = np.zeros(16, dtype=np.float32)

    assert validator(audio, "one two three", 48000) is True

    heard["text"] = "completely different words entirely"
    assert validator(audio, "one two three", 48000) is False

    assert [c["accepted"] for c in validator.calls] == [True, False]
    assert validator.calls[0]["wer"] == pytest.approx(0.0)


def test_missing_mlx_whisper_gives_an_actionable_error(monkeypatch):
    mod = _load()
    validator = mod.WhisperWerValidator("unused/repo")
    monkeypatch.setitem(__import__("sys").modules, "mlx_whisper", None)

    # `import mlx_whisper` raises ImportError when the cached module is None.
    with pytest.raises(SystemExit, match="pip install mlx-whisper"):
        validator.transcribe(np.zeros(16, dtype=np.float32), 48000)


def test_resampler_is_self_contained_and_sane():
    """The example resamples 48k -> 16k itself rather than importing a private helper."""
    mod = _load()
    source = EXAMPLE.read_text(encoding="utf-8")
    assert "from dots_tts_mlx.model import" not in source, (
        "an example must not reach into the library's private module surface; "
        "resample_mono() exists so this stays a public-API-only sample"
    )

    sample_rate = 48000
    t = np.arange(sample_rate) / sample_rate
    tone = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)

    out = mod.resample_mono(tone, sample_rate, 16000)
    assert out.shape == (16000,)
    assert out.dtype == np.float32
    # A 1 kHz tone is far inside both passbands, so amplitude must survive intact.
    assert np.sqrt(np.mean(out**2)) == pytest.approx(1 / np.sqrt(2), abs=0.02)
    # A 12 kHz tone is above the 8 kHz output Nyquist and must be filtered, not folded.
    alias = np.sin(2 * np.pi * 12000.0 * t).astype(np.float32)
    assert np.sqrt(np.mean(mod.resample_mono(alias, sample_rate, 16000) ** 2)) < 0.01
    # Degenerate inputs.
    assert np.array_equal(mod.resample_mono(tone, sample_rate, sample_rate), tone)
    assert mod.resample_mono(np.zeros(0, dtype=np.float32), sample_rate, 16000).size == 0


@pytest.mark.parametrize(
    "text",
    [
        "This is ordinary spaced English.",
        "Ein deutscher Satz mit Umlauten: schön, größer, Straße.",
        "यह हिन्दी वाक्य है जिसमें रिक्त स्थान हैं।",  # Devanagari IS space-delimited
        "한국어 문장은 띄어쓰기를 합니다.",  # Hangul is space-delimited
    ],
)
def test_whitespace_delimited_scripts_are_accepted(text):
    assert _load().unscorable_script(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "这是一个中文句子。",  # Chinese
        "これは日本語の文です。",  # Japanese
        "นี่คือประโยคภาษาไทย",  # Thai
        "Mostly English but with 中文 embedded.",  # one clause is enough to break WER
    ],
)
def test_no_space_scripts_are_refused_with_an_actionable_reason(text):
    """WER over `\\w+` degenerates to 0-or-1 here, so the example must refuse, not guess."""
    reason = _load().unscorable_script(text)
    assert reason is not None
    assert "whitespace-delimited" in reason
    assert "normalize_words" in reason  # tells the reader what to swap


def test_no_space_script_degeneracy_is_real():
    """The reason the guard exists: one token in, so WER can only be 0.0 or 1.0."""
    mod = _load()
    chinese = "这是一个中文句子"
    assert len(mod.normalize_words(chinese)) == 1
    assert mod.word_error_rate(chinese, chinese) == 0.0
    assert mod.word_error_rate(chinese, "这是另一个句子") == 1.0


def test_max_wer_rejects_a_negative_threshold():
    """A negative rate can never be met: every chunk would burn the full retry budget."""
    import argparse

    mod = _load()
    with pytest.raises(argparse.ArgumentTypeError, match=">= 0"):
        mod.wer_threshold("-0.1")
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(
            ["--text", "hi", "--ref-audio", "r.wav", "--ref-text", "t", "--max-wer", "-1"]
        )


def test_max_wer_above_one_is_allowed_on_purpose():
    """WER is unbounded above (insertions divide by the REFERENCE length), so >1 is real."""
    mod = _load()
    assert mod.wer_threshold("1.5") == 1.5
    assert mod.wer_threshold("0") == 0.0
    # A chunk that says the right words plus a fabricated sentence scores above 1.0,
    # which is exactly the region a >1 threshold chooses to tolerate.
    assert mod.word_error_rate("one two", "one two three four five six") > 1.0
