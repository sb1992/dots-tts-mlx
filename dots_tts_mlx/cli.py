"""dots.tts-MLX voice-cloning CLI — pure-MLX multilingual TTS.

A continuous autoregressive flow-matching model: clean utterance onset, no discrete-codec
warm-up mumble. Writes ``{out-path}/{out-prefix}_000.wav`` @ 48 kHz.

Runtime code imports only mlx / numpy / soundfile + ``dots_tts_mlx`` and makes no
torch calls. (torch + transformers are present transitively via mlx-lm — same as
the wider MLX-audio ecosystem — but the inference math is pure MLX.)

Usage:
    dots-tts --text "..." --ref-audio ref.wav \
        --ref-text "transcript of ref.wav" --out-prefix dots_clone --language EN

    # equivalently, as a module:
    python -m dots_tts_mlx.cli --text "..." --ref-audio ref.wav --language EN
"""
from __future__ import annotations

import argparse

import mlx.core as mx

mx.set_memory_limit(int(45 * (1 << 30)))  # memory-ceiling safety guard — set BEFORE heavy alloc

import os  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from dots_tts_mlx.loader import from_pretrained  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    ap = argparse.ArgumentParser(description="dots.tts-MLX voice-cloning TTS")
    ap.add_argument("--model", default="weights/dots_tts_mlx")
    ap.add_argument("--text", default=None)
    ap.add_argument("--ref-audio", default=None)
    ap.add_argument(
        "--ref-text",
        default=None,
        help="reference transcript (required with --ref-audio for cloning).",
    )
    ap.add_argument("--out-path", default="outputs/dots_tts")
    ap.add_argument("--out-prefix", default="dots_clone")
    ap.add_argument(
        "--language",
        default=None,
        help="uppercase ISO code (e.g. EN/DE/ES/FR/HI); default None = no tag.",
    )
    ap.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="solver steps. Default resolves per checkpoint: 4 (meanflow/mf) or 10 "
        "(flow-matching/soar). Pass an explicit value to override (e.g. NFE 2 for mf).",
    )
    ap.add_argument(
        "--guidance-scale",
        type=float,
        default=1.2,
        help="CFG scale (flow-matching only; ignored by the meanflow/mf checkpoint, "
        "which has CFG fused into the distilled student).",
    )
    ap.add_argument("--speaker-scale", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-generate-length", type=int, default=500)
    ap.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback tempo on the output wav via pitch-preserving ffmpeg atempo "
        "(<1 = slower, >1 = faster; 0.9 ~= 10%% slower). dots.tts is a self-pacing AR "
        "model with no native rate control, so this is a post-hoc time-stretch.",
    )
    ap.add_argument(
        "--trim-onset",
        dest="trim_onset",
        action="store_true",
        default=True,
        help="trim the fixed BigVGAN vocoder onset transient (a soft ~50-150 ms 'hhh'/breath at "
        "sample 0) via an energy gate + 10 ms fade-in. ON by default; applied BEFORE --speed.",
    )
    ap.add_argument(
        "--no-trim-onset",
        dest="trim_onset",
        action="store_false",
        help="disable onset-transient trimming (keep the raw vocoder output verbatim).",
    )
    ap.add_argument(
        "--enroll",
        action="store_true",
        help="enrollment mode: compute a reusable SpeakerProfile from --ref-audio/--ref-text.",
    )
    ap.add_argument(
        "--profile-out",
        default=None,
        help="(enroll mode) directory to write the .dtprofile bundle to.",
    )
    ap.add_argument(
        "--profile",
        default=None,
        help="(use mode) a .dtprofile to clone from — replaces --ref-audio/--ref-text.",
    )
    ap.add_argument(
        "--no-streaming-decode",
        action="store_true",
        help="Use the legacy recompute-full patch re-encode (O(n^3)); default is the "
        "O(n) streaming decode. Numerically identical; for A/B / debugging only.",
    )
    ap.add_argument(
        "--long",
        action="store_true",
        help="Sentence-chunked long-text generation: split into sentences, generate each "
        "separately, concatenate. Fixes long/multilingual EOS truncation + keeps gen linear.",
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Override the per-chunk character cap (--long). Default is script-aware.",
    )
    ap.add_argument(
        "--gap-ms",
        type=int,
        default=80,
        help="Silence (ms) inserted between sentence chunks in --long mode.",
    )
    ap.add_argument(
        "--no-retry-degenerate",
        dest="retry_degenerate",
        action="store_false",
        help="(--long) disable the per-chunk reseed-retry guard (default: on).",
    )
    ap.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="(--long) max reseed-retries per degenerate chunk (default 2; 0 = off).",
    )
    ap.add_argument(
        "--no-reuse-reference",
        dest="reuse_reference",
        action="store_false",
        help="(--long, with --ref-audio) re-encode the reference for EVERY chunk instead "
        "of enrolling it once up front. Default is to enroll once; the flag restores the "
        "old per-chunk encode for A/B / debugging.",
    )
    return ap


_build_parser = build_parser


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()

    if args.enroll:
        if not args.ref_audio or not args.ref_text or not args.profile_out:
            ap.error("--enroll requires --ref-audio, --ref-text, and --profile-out")
        model = from_pretrained(args.model, dtype=mx.bfloat16).model
        prof = model.enroll(args.ref_audio, args.ref_text, speaker_scale=args.speaker_scale)
        prof.save(args.profile_out)
        print(
            f"[dots] enrolled -> {args.profile_out}  "
            f"({prof.prompt_patch_count} prompt patches, scale {prof.speaker_scale})  "
            f"MLX peak {mx.get_peak_memory() / (1 << 30):.2f}GB",
            flush=True,
        )
        return 0

    if not args.text:
        ap.error("--text is required for generation")

    model = from_pretrained(args.model, dtype=mx.bfloat16).model

    if args.profile:
        from dots_tts_mlx.profile import SpeakerProfile

        prof = SpeakerProfile.load(args.profile)
        _gen = model.generate_long if args.long else model.generate
        _kw = (
            {"gap_ms": args.gap_ms, "max_chars": args.max_chars,
             "retry_degenerate": args.retry_degenerate, "max_retries": args.max_retries}
            if args.long else {}
        )
        out = _gen(
            text=args.text,
            profile=prof,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            language=args.language,
            seed=args.seed,
            max_generate_length=args.max_generate_length,
            trim_onset=args.trim_onset,
            streaming_decode=not args.no_streaming_decode,
            **_kw,
        )
    else:
        if not args.ref_audio:
            ap.error("--ref-audio is required (or pass --profile)")
        if not args.ref_text:
            ap.error(
                "--ref-text (the reference transcript) is required with --ref-audio. "
                "The x-vector-only no-transcript clone has been removed."
            )
        _gen = model.generate_long if args.long else model.generate
        _kw = (
            {"gap_ms": args.gap_ms, "max_chars": args.max_chars,
             "retry_degenerate": args.retry_degenerate, "max_retries": args.max_retries,
             "reuse_reference": args.reuse_reference}
            if args.long else {}
        )
        out = _gen(
            text=args.text,
            prompt_audio=args.ref_audio,
            prompt_text=args.ref_text,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            speaker_scale=args.speaker_scale,
            language=args.language,
            seed=args.seed,
            max_generate_length=args.max_generate_length,
            trim_onset=args.trim_onset,
            streaming_decode=not args.no_streaming_decode,
            **_kw,
        )

    # Onset-transient trimming now happens INSIDE generate() (default-on, gated by
    # trim_onset above) so every caller gets it; nothing to do here but write.
    wav = np.asarray(out["audio"].astype(mx.float32)).ravel()
    sr = int(out["sample_rate"])

    os.makedirs(args.out_path, exist_ok=True)
    out_file = os.path.join(args.out_path, f"{args.out_prefix}_000.wav")
    sf.write(out_file, wav, sr)

    if abs(args.speed - 1.0) > 1e-3:
        # Post-hoc pitch-preserving time-stretch — dots.tts has no native speech-rate knob
        # (self-pacing AR model). atempo accepts [0.5, 2.0] per filter; chain for extremes.
        factors, t = [], float(args.speed)
        while t > 2.0:
            factors.append(2.0)
            t /= 2.0
        while t < 0.5:
            factors.append(0.5)
            t /= 0.5
        factors.append(t)
        chain = ",".join(f"atempo={f:.6f}" for f in factors)
        tmp = out_file + ".tmp.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", out_file, "-filter:a", chain, tmp],
            check=True,
        )
        os.replace(tmp, out_file)
        wav = np.asarray(sf.read(out_file)[0]).ravel()
        print(f"[dots] applied --speed {args.speed} (atempo {chain})", flush=True)

    dur = wav.shape[-1] / sr
    print(
        f"[dots] -> {out_file}  {dur:.2f}s @ {sr}Hz  "
        f"{out.get('num_patches', '?')} patches  "
        f"MLX peak {mx.get_peak_memory() / (1 << 30):.2f}GB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
