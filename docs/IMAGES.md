# Images

New since the G13 publication (G15a-G16). DeepSeek-V4.1-Flash sees images: the EXL3 packs keep its ViT, and the
engine runs DeepSeek's tower and aligner (BF16, ~0.97 GB) **on rank 0 only**; the image rows reach rank 1 through
the embedding exchange both ranks already do. Image positions route experts with the release's `gate.bias_vl` and get
no Engram contribution, as in DeepSeek's reference (`inference/vision.py`, `image_processor.py`, `model.py`).

```bash
IMG=$(base64 -w0 photo.png)
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "DeepSeek-V4.1-Flash-TF",
  "reasoning_effort": "none", "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What does the sign in this photo say?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$IMG"'"}}]}]}'
```

**Config** (on in `config/prod.env.example`):

```
TF_DSV41_IMAGES=native                    # placeholder (engine default) | reject | native
TF_DSV41_BIAS_VL=/cache/dsv41-bias-vl     # folder holding bias_vl.safetensors, in the cache volume
TF_DSV41_VISION_PREP_MB=64                # preprocessed images cached by data: URL digest
```

**The image routing bias.** Both EXL3 2.9 bpw packs dropped the 43 `*.ffn.gate.bias_vl` tensors (40 layers + 3
DSpark blocks). `scripts/serve.sh prebuild` (or `scripts/serve.sh cache` alone) fetches them into the cache volume on
both nodes, 66 KB read by HTTP range requests from `deepseek-ai/DeepSeek-V4.1-Flash` (each node needs network access
once). By hand, inside the image with the volume mounted:

```bash
python -m tensorfold.families.deepseek_v41.cuda.bias_vl_fetch /cache/dsv41-bias-vl        # or --src <local release dir>
```

The file records the source revision and each tensor's SHA-256. Without it, native mode still runs and image rows
route with the text bias (what the vLLM kit does); the boot log says so. Users who do not want images on the GPU at
all set **`TF_DSV41_IMAGES=placeholder`** (the engine's default, no tower loaded, no bias needed): each image part
becomes a short notice telling the model it cannot see the image, so agent turns with screenshots still work;
`reject` answers HTTP 400 instead.

**Accepted and refused.** Image parts in user, tool and system messages: OpenAI `image_url` (a `data:` URL or a
public `https` URL, fetched with a 10 s / 30 s timeout), `input_image`, `image`, and Anthropic `source` blocks. At
most **8 images a request** (`TF_DSV41_VISION_MAX_IMAGES`; long agent histories should drop old screenshots), 20 MB
and 64 M pixels an image, at most 1,024 positions an image; `http`, private and link-local addresses are refused. A
`<｜deepseek_image｜>` token typed into text is escaped to plain text, never treated as an image.

**Measured** (G15b-G16): visual questions 20 / 20; text-only replies byte-identical with images on, off or
placeholder (4 / 4 HTTP text probes, and the speed suite's replies); hit == cold, batched == alone and turn-2 replay
exact with images in the prompt; 4 image requests during a 299K prefill all answered with the worker at >= 5.1 GiB.
Cost: the head's MemAvailable at boot -0.8 to -1.4 GiB. The duplicate-image bug (the same screenshot twice in one
request killed both ranks) is fixed in this engine (see [Changes since G13](CHANGELOG.md)).

## The Zig serving path

The [Zig engine](ZIG-SERVE.md) serves images the same way: the tower and aligner on rank 0, image rows routed with
`gate.bias_vl` on both ranks, no Engram contribution at image positions. Requests take the same image parts and the
same limits (8 images, 20 MB and 64 M pixels an image, 1,024 positions an image); its preprocessed-image cache
(`TF_DSV41_VISION_PREP_MB`) defaults to 256 MB. The [serving profile](../config/prod-zig.env.example) turns it on:

```
TF_DSV41_IMAGES=native
TF_DSV41_BIAS_VL=/cache/dsv41-bias-vl
```

`TF_DSV41_BIAS_VL` names a folder in the `CACHE_DIR` mount (`/cache` in `scripts/run-zig.sh`). The Zig image carries
no Python, so fetch `bias_vl.safetensors` with the Python serving image, once on each node into that node's
`CACHE_DIR`:

```bash
docker run --rm -v "$CACHE_DIR:/cache" -e HF_HUB_OFFLINE=0 --entrypoint python <python-serving-image> \
  -m tensorfold.families.deepseek_v41.cuda.bias_vl_fetch /cache/dsv41-bias-vl
```

If the variable names a folder without the tensors, boot fails. Unset it to serve images with the text bias; the boot
log then says the image routing bias is missing.

### Zig measurements

Pending: the Zig vision gate on two DGX Sparks.

| Check | Result |
| --- | --- |
| Vision tower vs DeepSeek's reference | pending |
| Visual questions | pending (x / 20) |
| Memory with images on (rank 0 MemAvailable at boot) | pending |
| Text decode speed with images on vs off | pending |
