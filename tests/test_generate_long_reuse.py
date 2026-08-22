"""``generate_long(reuse_reference=…)`` — enroll the reference once, clone every chunk.

Two layers. The plumbing tests are **weight-free**: the model is a bare ``DotsTtsModel``
shell with ``generate``/``enroll`` monkeypatched, so they assert how many times the
reference is encoded and what conditioning each chunk is handed, without loading the 2B
model. The numeric claim — that reuse leaves attempt-0 audio unchanged — cannot be made
weight-free, so ``test_reuse_reference_matches_the_per_chunk_path`` at the bottom carries
the same ``slow`` + weights markers as the profile-parity gate it inherits its tolerance
from (``test_enroll.py::test_profile_generate_matches_one_shot``).
"""
import os
import pathlib

import mlx.core as mx
import numpy as np
import pytest

from dots_tts_mlx.model import DotsTtsModel

CHUNKS = ["one two three four five", "six seven eight nine ten"]

W = pathlib.Path(os.environ.get("DOTS_TTS_WEIGHTS", "weights/dots_tts_mlx"))
REF = pathlib.Path(os.environ.get("DOTS_TTS_REF", "reference.wav"))
REF_TEXT = os.environ.get("DOTS_TTS_REF_TEXT", "this is the reference transcript")


def _make_model():
    m = object.__new__(DotsTtsModel)
    m.sample_rate = 48000
    return m


def _audio(seconds=2.0, std=0.1):
    n = int(seconds * 48000)
    return mx.array((np.random.default_rng(0).standard_normal(n) * std)[None].astype(np.float32))


def _install(monkeypatch, *, enroll_raises=None):
    """Record every generate() call's conditioning + count enroll() calls."""
    monkeypatch.setattr("dots_tts_mlx.chunking.split_for_generation", lambda text, **k: CHUNKS)
    seen = {"generate": [], "enroll": 0}
    sentinel = object()  # stands in for a SpeakerProfile

    def fake_enroll(self, prompt_audio, prompt_text, *, speaker_scale=1.5):
        seen["enroll"] += 1
        if enroll_raises is not None:
            raise enroll_raises
        seen["enroll_args"] = (prompt_audio, prompt_text, speaker_scale)
        return sentinel

    def fake_generate(self, chunk, **kw):
        seen["generate"].append(kw)
        return {"audio": _audio(), "num_patches": 10, "sample_rate": 48000}

    monkeypatch.setattr(DotsTtsModel, "enroll", fake_enroll, raising=True)
    monkeypatch.setattr(DotsTtsModel, "generate", fake_generate, raising=True)
    return seen, sentinel


def test_reference_is_enrolled_once_and_reused(monkeypatch):
    """N chunks -> ONE reference encode, and every chunk clones from that profile."""
    m = _make_model()
    seen, sentinel = _install(monkeypatch)

    out = m.generate_long("...", prompt_audio="ref.wav", prompt_text="the transcript")

    assert seen["enroll"] == 1                      # not once per chunk
    assert out["reused_reference"] is True
    assert len(seen["generate"]) == len(CHUNKS)
    for kw in seen["generate"]:
        assert kw["profile"] is sentinel
        # must be cleared: generate() rejects profile + prompt_* together.
        assert kw["prompt_audio"] is None and kw["prompt_text"] is None


def test_enroll_uses_the_callers_speaker_scale(monkeypatch):
    """speaker_scale is baked into the profile, so it must reach enroll()."""
    m = _make_model()
    seen, _ = _install(monkeypatch)

    m.generate_long("...", prompt_audio="ref.wav", prompt_text="t", speaker_scale=2.0)

    assert seen["enroll_args"] == ("ref.wav", "t", 2.0)


def test_opt_out_keeps_the_per_chunk_reference(monkeypatch):
    m = _make_model()
    seen, _ = _install(monkeypatch)

    out = m.generate_long(
        "...", prompt_audio="ref.wav", prompt_text="t", reuse_reference=False
    )

    assert seen["enroll"] == 0
    assert out["reused_reference"] is False
    for kw in seen["generate"]:
        assert kw["prompt_audio"] == "ref.wav" and kw["prompt_text"] == "t"
        assert kw["profile"] is None


def test_supplied_profile_is_not_re_enrolled(monkeypatch):
    """An already-enrolled profile passes straight through — no second encode."""
    m = _make_model()
    seen, _ = _install(monkeypatch)
    caller_profile = object()

    out = m.generate_long("...", profile=caller_profile)

    assert seen["enroll"] == 0
    assert out["reused_reference"] is False
    assert all(kw["profile"] is caller_profile for kw in seen["generate"])


def test_plain_tts_without_reference_never_enrolls(monkeypatch):
    m = _make_model()
    seen, _ = _install(monkeypatch)

    out = m.generate_long("...")

    assert seen["enroll"] == 0 and out["reused_reference"] is False


def test_failed_enroll_falls_back_to_the_reference_path(monkeypatch):
    """Reuse is an optimization: if enroll() can't run, the render still succeeds.

    ``enroll()`` raises for a model built without a compat hash (i.e. not via
    ``from_pretrained``); that must not turn a working render into a failure.
    """
    m = _make_model()
    seen, _ = _install(monkeypatch, enroll_raises=ValueError("no compat hash"))

    out = m.generate_long("...", prompt_audio="ref.wav", prompt_text="t")

    assert seen["enroll"] == 1
    assert out["reused_reference"] is False
    assert out["num_chunks"] == len(CHUNKS)
    for kw in seen["generate"]:
        assert kw["prompt_audio"] == "ref.wav" and kw["prompt_text"] == "t"


def test_unexpected_enroll_failure_is_not_swallowed(monkeypatch):
    """Only ValueError (the no-compat-hash signal) falls back; a real bug surfaces.

    An AttributeError inside enroll() means the model is malformed, not that reuse is
    unavailable — swallowing it would silently downgrade to the slow path and hide the
    defect.
    """
    m = _make_model()
    _install(monkeypatch, enroll_raises=AttributeError("boom"))

    with pytest.raises(AttributeError, match="boom"):
        m.generate_long("...", prompt_audio="ref.wav", prompt_text="t")


# --- the numeric claim (needs weights) ---------------------------------------------

PARITY_TEXT = "Hello there, this is the first sentence. And here is the second one."


@pytest.mark.slow
@pytest.mark.skipif(not W.exists(), reason=f"weights absent at {W} (set $DOTS_TTS_WEIGHTS)")
@pytest.mark.skipif(not REF.exists(), reason="reference wav absent (set $DOTS_TTS_REF)")
def test_reuse_reference_matches_the_per_chunk_path():
    """reuse_reference=True == reuse_reference=False on attempt 0, same seed.

    The regression gate for the default flipping on. Retries are disabled so both runs
    are pure attempt-0 renders (a retry reseeds the decode noise and is expected to
    diverge — and, with reuse on, deliberately no longer resamples the reference).
    ``max_chars`` forces the two-sentence text into exactly two chunks, so the shared
    enrollment is actually exercised across a boundary. Tolerance is the one used by
    ``test_enroll.py::test_profile_generate_matches_one_shot``, the profile-parity gate
    this equality rests on.
    """
    m = DotsTtsModel.from_pretrained(W, dtype=mx.bfloat16)
    kw = {
        "prompt_audio": str(REF),
        "prompt_text": REF_TEXT,
        "num_steps": 6,
        "guidance_scale": 1.2,
        "speaker_scale": 1.5,
        "language": "EN",
        "seed": 42,
        "max_chars": 45,
        "retry_degenerate": False,
    }

    reused = m.generate_long(PARITY_TEXT, reuse_reference=True, **kw)
    per_chunk = m.generate_long(PARITY_TEXT, reuse_reference=False, **kw)

    assert reused["reused_reference"] is True
    assert per_chunk["reused_reference"] is False
    assert reused["num_chunks"] == per_chunk["num_chunks"] == 2

    a = np.asarray(reused["audio"].astype(mx.float32)).ravel()
    b = np.asarray(per_chunk["audio"].astype(mx.float32)).ravel()
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.max(np.abs(a)) > 0.01, "reused render is silence — parity test is meaningless"
    assert np.max(np.abs(a - b)) < 1e-3, float(np.max(np.abs(a - b)))
