# Supported Model Profiles

Lepton currently has two practical test profiles.

## TinyStories-15M — fast regression fixture

- File: `models/tinystories.gguf`
- Size: ~19 MB
- Architecture: Llama
- Layers: 6
- Hidden: 288
- Heads: 6 / 6 KV (MHA)
- FFN: 768
- Vocabulary: 32,000
- Context: 128
- Weights: Q4_0/Q8_0/F32 norms
- Tokenizer: SentencePiece-style `▁`

Use it for:

- fast compiler/kernel regression;
- deterministic `selftest`;
- GQA and tokenizer fixtures;
- QEMU benchmarks.

Expected:

```text
fwdpos 0 1       -> 9038 (' Once')
selftest         -> PASS
```

## SmolLM2-135M — realistic model profile

- File: `models/smallm.gguf`
- Size: 91,726,304 bytes
- SHA-256: `6f96e054a48f8fcdb5814abc1203ca12c0f9cce2791ca9a5153caad8c3b5479b`
- Architecture: Llama
- Layers: 30
- Hidden: 576
- Heads: 9 / 3 KV (GQA)
- Head dimension: 64
- FFN: 1536
- Vocabulary: 49,152
- Context metadata: 8192; Lepton caps it to **256**
- Weights: Q4_0/Q8_0/F32 norms
- Tokenizer: GPT-2 byte-level BPE
- Tied token embedding/output
- Current file behaves as a base completion model; for Q&A use an Instruct variant
- RoPE base: 100000
- BOS = EOS = 0

Use it for:

- realistic completion quality;
- GQA validation with a real model;
- GPT-2 tokenizer validation;
- tied-embedding validation;
- ISO/model-load testing;
- measuring realistic latency.

Expected startup:

```text
config
  arch      = llama
  n_embd    = 576
  n_ff      = 1536
  n_layer   = 30
  n_head    = 9
  n_head_kv = 3
  head_dim  = 64
  max_seq   = 256
  vocab     = 49152
  inference = supported
  rope_base = 100000
```

Tokenizer check:

```text
encode Once upon a time
  [0] 6403 = 'Once'
  [1] 1980 = ' upon'
  [2] 253  = ' a'
  [3] 655  = ' time'
```

Generation check (base completion, not chat):

```text
genprompt "Once upon a time" 0 1 3
  , there was
```

`genprompt` is the reliable path for this base file. `ask` is implemented for
SmolLM2-Instruct ChatML and will repeat or ramble when given a base model.

`selftest` recognizes the SmolLM2 profile:

```text
selftest
  selftest PASS (SmolLM BOS 29, prompt 28)
```

QEMU latency measured on the reference host:

```text
bench 1
  bench: 1 tokens in 2440 ms (2440 ms/token)
```

S9 latency will be substantially higher. Use short prompts and 1–5 generated
tokens while developing.

## Why cap context at 256?

At `n_layer=30`, `n_head_kv=3`, `head_dim=64`, a 256-position KV cache is about
12 MB; the full workspace is about 16 MB. The kernel heap is capped at 64 MB
and the inference workspace budget is 60 MB. Larger contexts would fit the
budget at 512 positions but are not yet enabled.

## Why GPT-2 tokenization matters

TinyStories uses SentencePiece-style `▁` word markers. SmolLM2 uses GPT-2
byte-level BPE with `Ġ` space markers. The kernel now detects
`tokenizer.ggml.model = gpt2` and:

- maps bytes through the GPT-2 byte-to-Unicode table;
- applies the merge list;
- normalizes `Ġ` back to a space for display and stop matching.

This is a byte-level encoder, not a full GPT-2 regex pre-tokenizer. Simple
English prompts work; unusual contractions or punctuation may tokenize
differently from the original GPT-2 regex.

## Asking questions

`ask` builds the SmolLM2 Instruct ChatML prompt:

```text
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
```

It uses the exact `<|im_start|>` and `<|im_end|>` vocabulary tokens, greedy
sampling, and stops on `<|im_end|>`. It requires an **Instruct** GGUF.
The ChatML prompt is about 37 tokens; `ask` prints prefill progress and
`Ctrl+C` cancels prefill or generation. On the S9, the first answer may take
minutes because prefill is not streamed from the terminal.

```text
ask "What is 2+2?" 20
```

The current `smallm.gguf` is a base completion model. Asking it a question
produces text like:

```text
2+2+2=42
```

That is a model-profile limitation, not a kernel bug. Download a SmolLM2-135M
**Instruct** GGUF and use the same `ask` command for short answers. Keep the
base file for completion/tokenizer/GQA tests.

## Recommended test mix

| Purpose | Model |
|---|---|
| Fast CI/QEMU regression | TinyStories |
| Deterministic fixture checks | TinyStories/synthetic |
| Real completion quality | SmolLM2-135M |
| GQA real-model test | SmolLM2-135M |
| GPT-2 tokenizer test | SmolLM2-135M |
| Tied embeddings | SmolLM2-135M |
| S9 load/latency test | SmolLM2-135M |

## SmolLM2-135M-Instruct — Q&A profile

Use this instead of the base file when you want short answers:

- File: `models/smollm2-instruct-q4_0.gguf`
- Size: 91,726,912 bytes
- SHA-256: `f68203bfb98b1b1e8c64fac75fab10c4e36acac081609573b4df0fcc19c90dd9`
- Architecture: Llama, 30 layers, 576 hidden, GQA 9/3
- Weights: Q4_0/Q8_0/F32 norms (all supported)
- Tokenizer: GPT-2 byte-level BPE
- ChatML with `<|im_start|>` / `<|im_end|>`
- BOS=1, EOS=2
- RoPE base 100000
- Context capped to 256

Download:

```bash
curl -L --fail -o models/smollm2-instruct-q4_0.gguf \
  https://huggingface.co/QuantFactory/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct.Q4_0.gguf
```

Test:

```bash
make clean-kernel
make iso MODEL=models/smollm2-instruct-q4_0.gguf
```

```text
ask "What is 2+2?" 8
```

Observed on QEMU:

```text
  prefilling 37 tokens (Ctrl+C cancels)...
  .........
  prefill: 89920 ms
  2 + 2 = 4.
  answer: 17150 ms for 8 token(s)
```

The prefill dominates. On the S9 this can be many minutes; use Ctrl+C and
`keytest` to distinguish input failure from prefill waiting.

The Q4_K_S/Q4_K_M variants from that repository mix in Q5_0 tensors, which
the kernel does not support yet. Use Q4_0 for now.

`selftest` is a base-profile check; on the Instruct model it reports a skip.
Use `ask` for Instruct validation.

Use SmolLM2 as the **demo/model-compatibility target**, but keep TinyStories
as the **fast deterministic regression fixture**.
