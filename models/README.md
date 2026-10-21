# Models

`tinystories.gguf` is the fast reference model used by QEMU/ISO regression and
the verification tools. `smallm.gguf` is the realistic SmolLM2-135M profile;
see [../MODELS.md](../MODELS.md). It is **not tracked in Git** by default because it is a
large binary with its own distribution terms.

## Expected file

- Path: `models/tinystories.gguf`
- Size: 19,077,344 bytes
- SHA-256:

  ```text
  66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739
  ```

- Format: GGUF v3, Llama architecture
- Tensors: 57
- Layers: 6
- Embedding width: 288
- Attention heads: 6
- Head dimension: 48
- FFN width: 768
- Vocabulary: 32,000
- Context length: 128
- Quantization: Q4_0 layer/embedding weights, Q8_0 output weights, F32 norms

The model is commonly called "TinyStories-15M", but the GGUF stores about
24.4M tensor elements because both `token_embd.weight` and `output.weight` are
present. The memory arithmetic in `ROADMAP.md` uses the stored tensors.

## Verify a local copy

```bash
sha256sum models/tinystories.gguf
```

The output must match the hash above.

## Use it

```bash
make run-kernel MODEL=models/tinystories.gguf
make iso MODEL=models/tinystories.gguf
```

## Should it be committed?

Only if you have confirmed the model's license and are willing to store a
~19 MB binary in the repository. For most repositories, keep it out of Git and
document the download source and checksum, as this file does.

If you decide to track it, prefer Git LFS:

```bash
git lfs install
git lfs track "*.gguf"
git add .gitattributes
git add -f models/tinystories.gguf
git commit -m "Add TinyStories model via Git LFS"
```

Check GitHub's Git LFS storage and bandwidth quotas before doing this.


## SmolLM2-135M profile

- Path: `models/smallm.gguf`
- Size: 91,726,304 bytes
- SHA-256:

  ```text
  6f96e054a48f8fcdb5814abc1203ca12c0f9cce2791ca9a5153caad8c3b5479b
  ```

- Format: GGUF v3, Llama architecture
- Tensors: 272
- Layers: 30
- Embedding width: 576
- Heads: 9 / 3 KV (GQA)
- Head dimension: 64
- FFN width: 1536
- Vocabulary: 49,152
- Context metadata: 8192 (Lepton caps at 256)
- Quantization: Q4_0 linear layers, Q8_0 token embedding, F32 norms
- Tokenizer: GPT-2 byte-level BPE
- Tied output embedding
- BOS = EOS = 0

## Use the realistic profile

```bash
make run-kernel MODEL=models/smallm.gguf
make iso MODEL=models/smallm.gguf
```
