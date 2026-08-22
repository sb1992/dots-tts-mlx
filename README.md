# dots-tts-mlx

A pure-[MLX](https://github.com/ml-explore/mlx) port of [`rednote-hilab/dots.tts`](https://github.com/rednote-hilab/dots.tts) — multilingual zero-shot voice-clone text-to-speech, running natively on Apple Silicon.

dots.tts is a **2B-parameter, fully continuous, end-to-end autoregressive flow-matching** TTS model. Unlike discrete-codec TTS models that warm up from a quantized token stream, dots.tts is continuous AR — so the **first patch is already a crisp utterance onset**, with no warm-up mumble at sample 0. It clones a voice from a short reference clip (audio + transcript) and synthesizes into 24 languages.

Upstream ships two post-trained checkpoints and **this port supports both**: the standard `dots.tts-soar` (SCA) decoder for quality, and the distilled `dots.tts-mf` **MeanFlow** decoder — a few-step (NFE=4) variant that's **~2× faster per clip** (see [Two decoders](#two-decoders--soar-quality-and-meanflow-fast)).

This repo is a clean-room MLX reimplementation of the runtime: no PyTorch calls in the inference path, gated per-stage against the original PyTorch model.

> **Ready-to-run weights** are published on Hugging Face — [`shraey/dots-tts-mlx`](https://huggingface.co/shraey/dots-tts-mlx). **Download and run — no PyTorch and no conversion step.** Quantized int4 builds are **~2.4 GB vs the original ~9 GB (−73%)**, for both the `soar` (quality) and `mf` (fast) decoders. `pip install` the runtime, `hf download` the weights, and go — see [Weights](#weights).

## Scope — what this is / isn't

This is a **converted-weight MLX inference runtime** for the `dots.tts-soar` (SCA) checkpoint. It deliberately does **not** replicate upstream's full surface. It **is**:

- a from-scratch MLX port of the dots.tts inference math, numerically gated against the original PyTorch model;
- a CLI + Python API that synthesizes from a **local, already-converted** weights directory;
- a small **runtime addition not present upstream**: [enroll a voice once](#enroll-once-reuse-a-voice) and reuse it — the reference encode is paid once, then every generation runs at a lower memory peak (upstream re-encodes the reference on every call).

It is **not** a drop-in replacement for the upstream package. In particular, this runtime does **not**:

- **auto-download by HF repo id** — you point it at a local converted directory (see [Weights](#weights)); there is no hub fetch baked into the runtime;
- **auto-detect language or resolve language names** — pass an **explicit uppercase ISO code** (e.g. `HI`, `ES`, `EN`), not `auto_detect` or a spelled-out name;
- **normalize text** — there is no `--normalize-text`; feed already-normalized input;
- **support random / no-reference sampling** — a reference clip (`--ref-audio`) is required for the voice clone;
- **ship a Gradio app** — CLI + Python API only;
- **do fine-tuning or training** — inference only.

If you need any of those, use the upstream project: [code](https://github.com/rednote-hilab/dots.tts) · [model](https://huggingface.co/rednote-hilab/dots.tts-soar).

## What it is

- **Architecture:** Qwen2.5-1.5B-Base LLM backbone (BPE text, no phonemes) → AR flow-matching DiT head → 48 kHz AudioVAE with a BigVGAN-style causal decoder. A frozen CAM++ speaker embedding conditions the speaker identity.
- **Zero-shot voice clone:** one reference wav + its transcript clones the voice; cross-language transfer works from a single reference.
- **Clean onsets:** continuous AR means no discrete-token warm-up — the clip opens on a real word.
- **Two decoders:** `soar` (10-step, quality) and `mf` MeanFlow (NFE=4, **~2× faster**) — auto-detected from the weights; see [Two decoders](#two-decoders--soar-quality-and-meanflow-fast).

## Two decoders — `soar` (quality) and MeanFlow (fast)

Upstream dots.tts releases two post-trained checkpoints; this port runs both, selected purely by which weights directory you load (auto-detected from `config.json` — no flag):

| Decoder | Checkpoint | Sampler | Speed | Use |
|---|---|---|---|---|
| **soar** (SCA) | `dots.tts-soar` | 10-step flow-matching + CFG | baseline | default — best quality |
| **MeanFlow** | `dots.tts-mf` | **NFE=4, no CFG** (distilled) | **~2× faster** | latency-sensitive |

**MeanFlow** is a distilled few-step decoder: instead of ~10 Euler steps with classifier-free guidance, it predicts an average velocity and reaches full quality in **4 single-pass steps with no CFG**, cutting the dominant DiT cost. Measured here at **~1.9–2.2× faster** per clip on reference cloning (EN/HI/ZH); upstream's own benchmarks put MeanFlow NFE=4 **within ~1 SIM point of SCA** (essentially on par). It's a separate checkpoint (`soar` + a small `duration_embedder`); `--guidance-scale` is ignored in MeanFlow mode (CFG is fused into the student). The `soar` path is unchanged.

Both decoders clone the same way — **reference audio + transcript** (in-context). Get the weights for either under [Weights](#weights).

## Install

Requires Python ≥ 3.10 on Apple Silicon (MLX is Metal-only).

```bash
# quickest — install the published release directly:
pip install "git+https://github.com/sb1992/dots-tts-mlx.git@v0.3.1"

# or, for development (editable):
git clone https://github.com/sb1992/dots-tts-mlx.git
cd dots-tts-mlx
pip install -e .
```

The runtime deps are `mlx`, `mlx-lm`, `numpy`, `soundfile`, `tokenizers`. The `dots-tts` console command and `python -m dots_tts_mlx.cli` are both installed.

For **weight conversion** and the **dev parity oracle** (both use PyTorch), install the extra:

```bash
pip install -e '.[oracle]'   # torch, transformers, torchdiffeq, safetensors, librosa, torchaudio
```

Two distinct workflows use this extra — don't conflate them:

- **Weight conversion** (`python -m dots_tts_mlx.convert`) needs only `torch` + `safetensors` + `numpy` (a subset of `[oracle]`). It does **not** need the upstream package.
- **Regenerating parity fixtures** (`tools/oracle.py`) needs the full `[oracle]` extra **and** the upstream `dots_tts` package, which is **not on PyPI** — install it separately from source:

  ```bash
  pip install -e /path/to/dots.tts
  # or: pip install "git+https://github.com/rednote-hilab/dots.tts"
  ```

`ffmpeg` is needed only if you use `--speed` (post-hoc time-stretch).

## Weights

Two ways to get runnable MLX weights — **most people want Option A.**

### Option A — download ready MLX weights (recommended)

Pre-converted, pre-quantized MLX weights are published at
[`shraey/dots-tts-mlx`](https://huggingface.co/shraey/dots-tts-mlx). **No PyTorch, no conversion** —
download the variant you want and point the runtime at it:

```bash
# soar (quality), int4 — recommended (~2.4 GB, −73% vs the original ~9 GB)
hf download shraey/dots-tts-mlx --include "int4/*" --local-dir ./dots-tts-mlx-weights
dots-tts --model ./dots-tts-mlx-weights/int4 \
    --text "Hello from MLX." --ref-audio ref.wav --ref-text "transcript of ref.wav" --language EN

# MeanFlow (~2× faster) — same flow, the mf-* folders:
hf download shraey/dots-tts-mlx --include "mf-int4/*" --local-dir ./dots-tts-mlx-weights
dots-tts --model ./dots-tts-mlx-weights/mf-int4 \
    --text "Hello from MLX." --ref-audio ref.wav --ref-text "transcript of ref.wav" --language EN

# int8 / mf-int8 are the conservative fallbacks (~3.1 GB): swap the folder name above.
```

The downloaded folder is self-contained and loads exactly like an unquantized one — the runtime
auto-detects the `quantization` block (and, for `mf-*`, the `meanflow` block) in `config.json`, so
nothing changes at the CLI/API level.

**Variants** (pick a decoder × a precision):

| Folder | Decoder | Download | vs original |
|--------|---------|----------|-------------|
| original `dots.tts-soar` / `dots.tts-mf` (PyTorch) | — | ~9 GB | — |
| `int4/` ⭐ | soar — 10-step, quality | **~2.4 GB** | **−73%** |
| `int8/` | soar — 10-step, quality | ~3.1 GB | −65% |
| `mf-int4/` ⚡ | MeanFlow — NFE-4, **~2× faster** | ~2.4 GB | −73% |
| `mf-int8/` | MeanFlow — NFE-4, **~2× faster** | ~3.1 GB | −65% |

Only the **Qwen2.5 LLM trunk** (≈70% of the weights) is quantized; the precision-sensitive
flow-matching DiT, the BigVGAN vocoder, and the CAM++ speaker stay bf16. `int4` is the recommended
download (`mf-int4` if you want the faster decoder); `int8`/`mf-int8` are the conservative fallbacks.

**Quality.** Quantization is validated to be **lossless relative to the full-precision MLX build**: on a
small multilingual acceptance check (EN/DE/ES/FR + Hindi), int8 and int4 showed no transcription-accuracy
or voice-similarity regression vs bf16. This is a sanity check, **not a dataset-scale benchmark** —
evaluate on your own content. (Those 5 are only the quant-validation subset; the model supports all
**24 languages** of upstream dots.tts.) (Correctness of the port itself is gated per-stage against the original
PyTorch model — see [How it was ported / parity](#how-it-was-ported--parity).)

> **Why no bf16 download?** bf16 is the runtime dtype (the full-precision reference), but it showed no
> quality advantage over int4 in our checks at ~2× the size — so we don't host it. Produce it locally
> with `--bits 16` (Option B) if you want it.

### Option B — convert from source (advanced)

For reproducibility, re-quantizing, or auditing, convert the original checkpoint yourself (needs the
`[oracle]` extra for torch):

```bash
# 1. Download the original checkpoint (~9 GB).
hf download rednote-hilab/dots.tts-soar --local-dir weights/dots_tts_src/dots.tts-soar

# 2. Convert HF -> MLX fp32 safetensors.
python -m dots_tts_mlx.convert --src weights/dots_tts_src/dots.tts-soar --out weights/dots_tts_mlx

# 3. (optional) Quantize the LLM trunk — --bits {16,8,4}  (16 = bf16, no quantization; --group-size 64)
python -m dots_tts_mlx.quantize --src weights/dots_tts_mlx --out weights/dots_tts_mlx_int4 --bits 4
```

`convert` folds the vocoder's `weight_norm` (80 pairs), passes the speaker BN buffers through, extracts
`latent_stats`, and copies the config + tokenizer. Output is fp32 (~9 GB); the loader casts to bf16 at
runtime. `quantize` needs only `mlx` + `mlx-lm` (no torch). Only the converted/quantized artifacts are
needed to run — the original checkpoint + `[oracle]` extra can be removed afterward.

### MeanFlow weights from source (`mf-*`)

Ready-to-run `mf-int4` / `mf-int8` are in [Option A](#option-a--download-ready-mlx-weights-recommended) above; see [Two decoders](#two-decoders--soar-quality-and-meanflow-fast) for what MeanFlow is. To build them yourself, convert + quantize the upstream `dots.tts-mf` checkpoint — **exactly like soar** (the quantizer targets only the LLM trunk, so the `duration_embedder` stays bf16 and MeanFlow mode survives every variant):

```bash
hf download rednote-hilab/dots.tts-mf --local-dir weights/dots_tts_src/dots.tts-mf
python -m dots_tts_mlx.convert  --src weights/dots_tts_src/dots.tts-mf --out weights/dots_tts_mlx_mf
python -m dots_tts_mlx.quantize --src weights/dots_tts_mlx_mf --out weights/dots_tts_mlx_mf_int4 --bits 4  # or --bits 8 / 16
dots-tts --model weights/dots_tts_mlx_mf_int4 --text "..." --ref-audio ref.wav --ref-text "transcript of ref.wav" --language EN
```

MeanFlow mode is **auto-detected** from `config.json` (the `meanflow` block) — no flag. `--num-steps` then defaults to 4 (the NFE) and `--guidance-scale` is ignored (CFG is fused into the distilled student).

## CLI usage

```bash
dots-tts \
    --text "Hello, this is a quick test of the on-device voice." \
    --ref-audio path/to/reference.wav \
    --ref-text "transcript of the reference clip" \
    --language EN \
    --out-path outputs/dots_tts \
    --out-prefix my_clone
# -> outputs/dots_tts/my_clone_000.wav  (48 kHz)
```

Key flags:

- `--ref-audio` **+** `--ref-text` (both required to clone) — the voice to clone and its **exact transcript** (what is actually spoken in the clip). The transcript conditions the in-context clone; it is **never** part of the output — only your `--text` is synthesized.
- `--language` — uppercase ISO code (`EN` / `DE` / `ES` / `FR` / `HI` / …). Default `None` = no language tag.
- `--num-steps 10` `--guidance-scale 1.2` `--speaker-scale 1.5` `--seed 42` — flow-matching sampler knobs (defaults are the validated ship config).
- `--speed 1.0` — adjust playback tempo, **pitch-preserving** (ffmpeg `atempo`; `<1` slower, `>1` faster), applied after onset-trim.
- `--trim-onset` / `--no-trim-onset` — `--trim-onset` is **on by default**: it removes the fixed ~50–150 ms BigVGAN vocoder onset transient (a soft "hhh"/breath at sample 0) via an energy gate + 10 ms anti-click fade. `--no-trim-onset` keeps the raw vocoder output verbatim.
- `--no-streaming-decode` — the patch encoder re-encodes each generated patch incrementally (maintained conv tail + per-layer KV caches), which is **O(n)** instead of the legacy recompute-full's O(n²)-total. Streaming is **on by default** and numerically identical to recompute-full (the encoder is fully causal, no rotary/qk-norm); `--no-streaming-decode` selects the recompute-full fallback for A/B / debugging.

## Choosing a reference

Cloning is **in-context**: pass the reference audio (`--ref-audio`) **and its exact transcript**
(`--ref-text`) — what is actually spoken in the clip — and the model continues that prefix in the
cloned voice. The transcript is **never** part of the output; only your `--text` is synthesized.

**Keep the reference short (a few seconds).** A longer reference does **not** improve adherence,
and it costs noticeably more time and memory — the model re-attends the whole reference on every
step (and on every sentence under `--long`). A short, accurate transcript gives the closest match
at the lowest cost. For long cloned passages, [enroll the voice once](#enroll-once-reuse-a-voice)
so the reference encode isn't recomputed per sentence.

## Long / multilingual text (`--long`)

dots.tts is autoregressive, so **generation cost grows as the clip gets longer** (each new
moment attends over everything generated so far), and on long input the model can also **stop
early or drift** — most visibly in non-English languages. `--long` addresses both by generating
**one sentence at a time** and stitching the results:

```bash
dots-tts --text "First sentence. Second sentence. ..." \
    --ref-audio ref.wav --ref-text "transcript of ref.wav" --language EN --long

# non-Latin works too (splits on 。 ！ ？ for CJK, । for Devanagari):
dots-tts --text "नमस्ते दोस्तों। यह एक लंबा वाक्य है। धन्यवाद।" \
    --ref-audio ref.wav --language HI --long
```

It splits the text into sentences — a **word-safe** length cap sub-splits any over-long
sentence (never mid-word or mid-character) — generates each chunk independently, and
concatenates with a short silence gap (`--gap-ms`, default 80; `--max-chars` overrides the
per-chunk cap). Because every chunk stays short:

- **No truncation or drift** — each sentence gets a fresh, in-range context, so the whole
  passage is spoken, in any language.
- **Cost stays linear in length** instead of ballooning — so long passages stay tractable and
  are modestly quicker than one long pass. (It is *not* a per-clip speed-up — for that, use the
  [MeanFlow decoder](#meanflow-few-step-decoder-the-mf-checkpoint).)

**Reference cost under `--long`.** Each chunk re-applies the reference, and that prefix is
re-attended for *every* sentence — so for long text keep the reference **short**, and/or
[enroll the voice once](#enroll-once-reuse-a-voice) so the reference encode isn't recomputed per
sentence. `--speed` and `--profile` both work with `--long`.

> **Self-healing chunks (v0.5.1).** Under `--long`, each sentence chunk is health-checked
> (finite, non-silent, not absurdly short for its text); a degenerate chunk is regenerated
> with a fresh seed (default up to 2 retries). Disable with `--no-retry-degenerate`, tune
> with `--max-retries N`. Healthy chunks are unchanged (retries only fire on failure). The
> cheap guard catches truncation; callers with ASR can pass `generate_long(validator=…)` to
> also catch same-length hallucinations (the CLI stays dependency-free) — see
> [ASR-validated chunks](#asr-validated-chunks-example).

### ASR-validated chunks (example)

`chunk_health` cannot tell a correct chunk from one that is the right length but says the
wrong words — that needs a transcript, i.e. an ASR dependency the package deliberately
does not take. Instead, `generate_long(validator=…)` takes a
`(audio, chunk_text, sample_rate) -> bool` callback; returning `False` reseeds and
regenerates that chunk (up to `max_retries`).

[`examples/asr_validated_long.py`](examples/asr_validated_long.py) is a ready-to-run
implementation: transcribe each chunk with [mlx-whisper](https://pypi.org/project/mlx-whisper/),
accept it when its word error rate against the intended text is `<= --max-wer`.

```bash
pip install mlx-whisper   # not a dependency of dots-tts-mlx

python examples/asr_validated_long.py \
    --model weights/dots_tts_mlx \
    --text "First sentence. Second sentence. Third one here." \
    --ref-audio reference.wav --ref-text "transcript of reference.wav" \
    --language EN --max-wer 0.5
```

It uses WER rather than word *coverage* on purpose: coverage is recall-only, so a clip
that says everything it should **plus** a sentence of invented text still scores perfectly.
WER counts insertions and substitutions, which is the failure the hook exists to catch.
For the same reason `--max-wer` is deliberately **not** capped at 1.0 — insertions divide
by the *reference* length, so WER is unbounded above and a threshold above 1 is a real
(lax) setting. Negative values are rejected: they can never be met, so every chunk would
burn its whole retry budget.

> **The example is scoped to whitespace-delimited languages.** Its tokenizer is `\w+`,
> which is unicode-aware but still effectively whitespace-delimited: in a script written
> without spaces (Chinese, Japanese, Thai, Lao, Khmer, Burmese) a whole clause becomes one
> token, so WER can only come out 0.0 or 1.0 and the gate carries no information. The
> script detects those up front and exits with an explanation rather than reporting a
> meaningless number. dots.tts renders those languages fine — it is this *example's metric*
> that does not apply; gating them needs a language-aware segmenter (jieba / MeCab /
> PyThaiNLP) in place of `normalize_words`.

## Python API

```python
import mlx.core as mx
from dots_tts_mlx.loader import from_pretrained

model = from_pretrained("weights/dots_tts_mlx", dtype=mx.bfloat16).model

out = model.generate(
    text="Hello from MLX.",
    prompt_audio="reference.wav",
    prompt_text="transcript of reference.wav",   # the reference transcript (required to clone)
    language="EN",
    num_steps=10,
    guidance_scale=1.2,
    speaker_scale=1.5,
    seed=42,
)

wav = mx.array(out["audio"]).astype(mx.float32)   # mono float32
sr = out["sample_rate"]                            # 48000
```

## Enroll once, reuse a voice

> **v0.5.0 — re-enroll required.** v0.5.0 corrects the in-context prompt-embedding latent
> scale: the patch encoder is now fed *denormalized* reference latents (matching upstream),
> fixing reference conditioning on the `--ref-text` / `prompt_text` path. Profiles enrolled
> on ≤ v0.4.x (`schema_version` 1) cached the old (normalized-derived) embedding and are
> **rejected on load by design** — just re-run `--enroll` to regenerate them. Model weights
> are unchanged.

Compute a voice's reference conditioning **once**, save it to disk, and reuse it for every
later generation — so you never re-pass the reference, and the expensive reference encode
(the CAM++ speaker embedding + the AudioVAE encode of the reference + the patch-encoder pass) is paid
**once at enrollment** instead of on every call.

```bash
# 1. enroll a voice -> a reusable .dtprofile bundle
dots-tts --enroll --ref-audio reference.wav \
    --ref-text "transcript of reference.wav" --profile-out alice.dtprofile

# 2. generate from the profile — no --ref-audio / --ref-text needed
dots-tts --profile alice.dtprofile --text "Hello from the enrolled voice." \
    --language EN --out-prefix clone
```

```python
profile = model.enroll("reference.wav", "transcript of reference.wav", speaker_scale=1.5)
profile.save("alice.dtprofile")                       # cond.safetensors + profile.json (<2 MB)

from dots_tts_mlx.profile import SpeakerProfile
profile = SpeakerProfile.load("alice.dtprofile")
out = model.generate("Hello from the enrolled voice.", profile=profile, language="EN")
```

- **Footprint:** profile generation skips the reference re-encode every call, dropping the
  steady-state peak from ~10.8 GB to **~6.6 GB** (−39%). The
  ~10 GB enrollment peak is paid once.
- **Output is identical** to the equivalent one-shot `generate(prompt_audio=…, prompt_text=…)`.
- **Portable across precisions:** a profile enrolled on int4 also loads on int8 / bf16
  (the cached conditioning comes from bf16-only components). Loading against a different
  model raises a clear error.
- **Pairs with `--long`:** chunked long-form generation otherwise re-encodes the reference
  **once per sentence**. A profile does that work **once**, so enrolling and passing `--profile`
  is the efficient way to clone a voice across a long passage.
- `--enroll` requires `--ref-text`; `--profile` is mutually exclusive with `--ref-audio`/`--ref-text`.

> **Why this exists / not in upstream.** Upstream `dots.tts` has no enroll/profile concept — it
> re-encodes the reference (the CAM++ speaker embedding + the AudioVAE encode + the patch-encoder pass) on **every**
> `generate`, and that encode is the ~10 GB memory high-water. This is a thin **runtime/app-layer**
> addition for Apple Silicon: do that work once, persist the small result (<2 MB), and skip it on every
> later call — so steady-state generation fits in **~6.6 GB instead of ~10.8 GB** and is faster, with
> **bit-identical** output. It does not change the model or the inference math.

## Roadmap

- **Cheaper cloned chunking.** Reusing one enrolled reference across `--long` chunks (so the
  in-context prefix isn't re-attended per sentence) is a planned optimization; today, keep the
  reference short or [enroll the voice once](#enroll-once-reuse-a-voice) for long cloned passages.

## How it was ported / parity

Every stage was gated numerically against the original PyTorch model (a dev-only oracle, `tools/oracle.py`, dumps reference fixtures) before any behavioral test. Each gate uses a manual high-precision fp32 reference matmul so a single component's parity is isolated from the runtime's reduced-precision matmul:

| Stage | Metric | Result |
|-------|--------|--------|
| AudioVAE decode | PSNR vs torch | 55.67 dB |
| AudioVAE encode | max-abs | 2.3e-4 |
| CAM++ speaker embedding | cosine | 0.99999988 |
| Attention (rotary) | max-abs | 4.77e-7 |
| DiT velocity field | cosine | 0.9999962 |
| Semantic patch encoder | cosine / max-abs | 0.9999995 / 4.69e-4 |
| LLM hidden (Qwen2.5) | cosine | 0.99999970 |
| Flow solver (true-fp32 floor) | max-abs | 1.4e-4 |

**The tf32 finding — why the end-to-end test is behavioral, not sample-exact.** MLX's fast matmul rounds fp32 operands to ~tf32 (10-bit mantissa) on the GPU. The per-stage gates sidestep this with an explicit high-precision path. But the euler ODE in the flow solver *amplifies* the per-step DiT matmul tf32 floor (fast-path max-abs 0.577 vs the true-fp32 floor of 1.4e-4), so across 10 integration steps × N patches the trajectory diverges enough that the waveform does **not** sample-align with the PyTorch golden. This is a runtime-precision property, not a port bug — sample-exact e2e parity isn't reachable on this hardware path. So the e2e gate is **behavioral**: the output is intelligible, in the correct language, voice-matched, finite, the right duration, and clean-onset. The component-level numerics prove the math; the behavioral gate proves the product.

Tests live in `tests/` (pytest). They skip cleanly when the converted weights or PyTorch fixtures are absent. Regenerate fixtures with `tools/oracle.py` under the `[oracle]` extra **plus** the upstream `dots_tts` package installed from source (`pip install -e /path/to/dots.tts`) — see [Install](#install).

## Requirements & notes

These are specifics of *this MLX port* — not limitations of the model itself, which behaves the same as upstream dots.tts.

- **Apple Silicon only** — MLX is Metal-only (the upstream PyTorch model targets CUDA).
- **Footprint:** peak RAM scales with generation + reference length — roughly **~6 GB for a short clip, up to ~13 GB for a ~30 s clip** (int4; resident weights ~2.4 GB). The render peak is **activation-bound** (the bf16 DiT + vocoder working set), so it's the same for soar and MeanFlow and is **not** reduced by quantization (quant shrinks the resident floor, not the transient peak). Inherent to the 2B model (same class as upstream); fp32 runs land ~2×. MLX's allocator may cache up to its limit, but that cache is releasable.
- The runtime makes **no torch calls**, though `torch` / `transformers` are pulled in transitively by `mlx-lm`'s tokenizer utilities — the inference math is pure MLX.

## Responsible use

This runtime performs **zero-shot voice cloning** — it can reproduce a person's voice from a few seconds of reference audio. That capability carries real risk of misuse. By using this software you agree to use it responsibly. Mirroring the upstream [dots.tts-soar risks guidance](https://huggingface.co/rednote-hilab/dots.tts-soar):

- **No impersonation, fraud, or disinformation.** Do not use cloned voices to impersonate real people without authorization, to commit fraud or social engineering, to evade voice-based authentication, or to produce misleading or deceptive content.
- **Consent for reference audio.** Only clone a voice you own or for which you have the speaker's explicit, informed consent. Respect the rights of voice owners and applicable privacy / publicity / data-protection laws in your jurisdiction.
- **Disclose AI-generated audio.** Clearly label synthesized speech as AI-generated wherever it is published or shared, so listeners are never misled about its origin.
- **Watermark + detect downstream.** You are encouraged to apply audio watermarking to generated output and to deploy synthetic-speech detection in any pipeline that ingests it, to support provenance and abuse mitigation.

The authors and contributors disclaim responsibility for misuse. Comply with all applicable laws and with the upstream model's license and usage terms.

## Attribution + licenses

This is a derivative port. The original model, its backbone, and the components it builds on are each independently licensed (all verified):

- **dots.tts** (`rednote-hilab/dots.tts-soar`) — **Apache-2.0**. The model, weights, and the *dots.tts Technical Report* (2026) are by the dots.tts team at rednote-hilab. [Model](https://huggingface.co/rednote-hilab/dots.tts-soar) · [Code](https://github.com/rednote-hilab/dots.tts) · [Demo](https://rednote-hilab.github.io/dots.tts-demo/)
- **Qwen2.5-1.5B-Base** (LLM backbone) — **Apache-2.0** ([Qwen](https://huggingface.co/Qwen/Qwen2.5-1.5B)).
- **CAM++ / 3D-Speaker** (speaker-embedding encoder) — **Apache-2.0** ([modelscope/3D-Speaker](https://github.com/modelscope/3D-Speaker)).
- **BigVGAN** (the vocoder/decoder architecture style) — **MIT**, © NVIDIA ([NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN); parts adapted from [hifi-gan](https://github.com/jik876/hifi-gan), MIT).

The MLX port code in this repository is licensed **Apache-2.0** (see [LICENSE](LICENSE)). You must comply with the upstream licenses for the model weights and any redistributed components.

### Credit

Full credit to the **dots.tts team at rednote-hilab** for the model, the SCA post-training, and the open release. This repo only re-expresses their runtime in MLX; the research, training, and weights are theirs.
