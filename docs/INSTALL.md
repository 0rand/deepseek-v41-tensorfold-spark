# Install and run

For the recommended q28-v2 pack and CUDA Zig path, see [ZIG-SERVE](ZIG-SERVE.md). The instructions below retain the published Python 0.6.0 recipe and its compatible 2.9 bpw pack.

Requirements:

- two DGX Sparks with their CX7 ports cabled and addressed (one link subnet), Docker with the NVIDIA runtime on both,
  and passwordless ssh from the head to the worker over the link;
- on each node's local NVMe: ~100 GB for the weights, ~95 GB for the Engram shards, ~95 GB for the prepared
  folders, and room for the session tier (`TF_DSV41_SESSION_DISK_GIB`, 128 GB by default);
- nothing else on the GPUs: the stack plans for a 4-5 GiB MemAvailable floor out of 128 GB a node.

**1. Clone and configure** (on the head):

```bash
git clone --recurse-submodules https://github.com/jayleaton/deepseek-v41-tensorfold-spark.git
cd deepseek-v41-tensorfold-spark
cp config/prod.env.example config/prod.env
$EDITOR config/prod.env      # every <placeholder>: WORKER_SSH, HEAD_IP, the paths on each node; check the NIC names
```

**2. Weights** (on both nodes, byte-identical):

```bash
hf download dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw --local-dir <HEAD_MODEL>    # and <WORKER_MODEL>
```

**3. Engram shards.** The EXL3 packs do not carry the Engram tables (layers 1 and 14, ~101 GB each). They come from
DeepSeek's original checkpoint; each rank keeps its half of the hash heads (~47 GiB a layer) on local NVMe:

```bash
mkdir -p <src> && hf download deepseek-ai/DeepSeek-V4.1-Flash model.safetensors.index.json --local-dir <src>
python3 scripts/pack_engram.py --src <src> --list                 # the shard files that hold the tables
hf download deepseek-ai/DeepSeek-V4.1-Flash <those files> --local-dir <src>
python3 scripts/pack_engram.py --src <src> --config <HEAD_MODEL>/config.json --out <HEAD_ENGRAM> --rank 0
python3 scripts/pack_engram.py --src <src> --config <HEAD_MODEL>/config.json --out <dir> --rank 1   # copy to <WORKER_ENGRAM> on the worker
python3 scripts/pack_engram.py --src <src> --config <HEAD_MODEL>/config.json --check <HEAD_ENGRAM> --rank 0
```

`pack_engram.py` writes the format the engine reads (`engram-l{1,14}-r{rank}of2.bin`) from the source tensors'
raw bytes; it is tested against the engine's reader on synthetic tables. The measured runs used shards packed by
the MiaAI-Lab kit (`./start.sh pack`), which writes the same format; `--check` compares either with the source.

**4. Build, check, start:**

```bash
scripts/serve.sh build        # docker/Dockerfile: TensorFold v0.6.0 + patches/, shipped to the worker, then prebuild
scripts/serve.sh preflight    # image on both nodes, weights, Engram shards, RoCE ports, free ports, idle GPUs
scripts/serve.sh start        # memory gate, rank 1 then rank 0, /v1/models, slot check, canary
```

`build` ends with `scripts/serve.sh prebuild`: the CUDA extensions (18, the G13-G17 kernels included) are compiled
into the `CACHE_VOL` volume on both nodes with no weights loaded, so no extension is built beside the weights. It then
copies the measured dense-prefill tuning table (`config/pfdense-table.json`) into the volume and, for native image
input, fetches the image routing bias the EXL3 packs dropped (66 KB of `deepseek-ai/DeepSeek-V4.1-Flash` by HTTP range
requests: both nodes need network access once; `scripts/serve.sh cache` repeats this step alone). Run it again after
clearing the volume. The first start compiles the Triton kernels and writes the prepared rank folders
(`TF_DSV41_PREPARED_WRITE=1`, ~95 GB a node, several minutes); later starts read them back in ~40 s. Then:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "DeepSeek-V4.1-Flash-TF",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}], "reasoning_effort": "low"}'
scripts/serve.sh status | logs [0|1] | canary | stop | restart
```

**The 1017 release.** [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
has the same architecture and config as V4.1-Flash, with new weights. These steps serve it unchanged, with four
differences:

- The pack is [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw),
  whose tensors match Mia-AiLab's 2.9 bpw pack.
- Its Engram tables changed. Pack them from the 1017 repo's shards 47 and 48 (`pack_engram.py --list` names them).
- It ships no chat template, and the server stops without one. `encoding/encoding.py` is byte-identical in the two
  releases, so Mia-AiLab's `chat_template.jinja` applies: copy it into the pack on both nodes.
- `serve.sh cache` fetches the image routing bias from the older release when none is there. Fetch the 1017 one first:
  `python -m tensorfold.families.deepseek_v41.cuda.bias_vl_fetch $TF_DSV41_BIAS_VL --repo deepseek-ai/DeepSeek-V4.1-Flash-1017`
  in the image, on both nodes.

On two Sparks it measured 100.6 tok/s on code at one stream and 155.8 at four (recipe defaults), with
[BobClawblaw/deepseek-v41-flash-1017-2x-dgx-sparks](https://github.com/BobClawblaw/deepseek-v41-flash-1017-2x-dgx-sparks)
scripting these steps.

**5. Run it unattended** (optional): a watchdog tick every minute (heals after 3 bad ticks, at most every 30 min; a
fail-fast exit, code 70, heals on the first tick, at most every 2 min) and a start at boot.

```bash
mkdir -p ~/.config/systemd/user && cp scripts/systemd/dsv41-* ~/.config/systemd/user/
$EDITOR ~/.config/systemd/user/dsv41-*.service        # WorkingDirectory= this checkout
loginctl enable-linger "$USER"
systemctl --user daemon-reload && systemctl --user enable --now dsv41-tf-watchdog.timer && systemctl --user enable dsv41-boot-start.service
```

`scripts/serve.sh` drops the page cache on both nodes around a start (`DROP_CACHES=1`: needs `sudo -n` or root for
`/proc/sys/vm/drop_caches`; otherwise it logs and continues). [`docs/OPERATIONS.md`](OPERATIONS.md) covers the
knobs, the memory gates, the watchdog, and how to turn each lever off.
