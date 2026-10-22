# API

OpenAI-compatible on `HOST:PORT` (`127.0.0.1:8000` by default: put your own proxy and authentication in front):
`/v1/chat/completions` (streaming, tool calls, `response_format`), `/v1/completions` (text or token ids),
`/v1/messages` (Anthropic's Messages format, streaming included, plus `/v1/messages/count_tokens`),
`/tokenize`, `/v1/models` (`max_model_len`), `/health`, `/metrics`.

`/v1/messages` streams follow the Anthropic streaming convention: `message_start` names the whole prompt
(`input_tokens`, output 0) and `message_delta` splits that same prompt into `input_tokens`,
`cache_read_input_tokens` and `cache_creation_input_tokens`. The head's number equals the delta's parts
added together. A context-window refusal opens with 0, since no prompt stands behind that head. The counts
need no request option: on this server a stream names its usage on its last chunk, asked for or not.

A DeepSeek added token quoted inside a chat request (`</think>`, `<think>`, `<｜User｜>`, `<｜Assistant｜>`,
`<｜end▁of▁sentence｜>`, `<｜DSML｜>`, the tool-call markers; 59 literals read from `tokenizer.json` at start) is
text to the model, never the token (`patches/0003`): a tool result or a pasted log that contains `</think>` cannot end
the reasoning or inject a turn boundary. The literals are hidden before the chat template renders, so the template's
own tokens are untouched; the `/v1/completions` route sends what it is given.

Thinking follows DeepSeek-V4.1's encoding (`Reasoning Effort: N (range 1-100)`), on by default
(`TF_DSV41_THINKING=0` turns the default off; requests can still turn it on). Both the top-level `reasoning_effort` and
`chat_template_kwargs.reasoning_effort` are read (the kwargs win):

| value | thinking | effort |
| --- | --- | ---: |
| `none`, `minimal` | off | - |
| `low` | on | 50 |
| `medium`, `high` (default: `TF_DSV41_DEFAULT_EFFORT`) | on | 75 |
| `xhigh`, `max` | on | 100 |
| an integer 1-100 | on | that |

`chat_template_kwargs.enable_thinking` (or `thinking`) true / false sets the mode directly. Note: the kit's vLLM
path renders `low` as 25; we keep DeepSeek's 50.

Effort sets how long the model thinks, not how fast it decodes: a lower effort gives a shorter `reasoning_content`
and an earlier answer at the same tok/s.

**Thinking and `max_tokens`.** Thinking is on by default at effort `high` (75), and `max_tokens` (or
`max_completion_tokens`) counts the `reasoning_content` and the `content` together. A short cap can therefore end the
reply while the model is still thinking: `content` comes back empty with `finish_reason: "length"`, and the Zig server
logs a warning. Raise `max_tokens`, or turn thinking off for the request with
`"chat_template_kwargs": {"enable_thinking": false}`, or for the server with `TF_DSV41_THINKING=0`.

## Sampling

Requests take the usual OpenAI sampling fields — `temperature`, `top_p`, `top_k`, `min_p`, `seed` — and
a request's value always wins. Anything a request omits falls back to the **server's default sampling**;
when a deployment sets none, the engine's fallback is `temperature 1.0, top_p 0.95, top_k 20, min_p 0`,
so an unconfigured server *samples* rather than decoding greedily. The `serve` command's `--temperature`,
`--top-p`, `--top-k` and `--min-p` set that server-side default.

Decoding is **greedy** when the effective `temperature` is `<= 0` — either the request asks for it or the
server default is greedy. Sampling is *keyed*: without an explicit `seed`, the seed is derived from the
prompt, so a sampled reply is reproducible for the same request — do not read "identical twice" as greedy.

One caution, because nothing in this project's own testing would surface it (the canary, the README cells
and the MMLU runs are all greedy): greedy is not free on long agentic contexts. A client that omits
`temperature` — most do — and inherits a greedy server default (`--temperature 0`) can spend its whole
reply budget looping inside `reasoning_content` and return an empty `content` with
`finish_reason: "length"`. The model card asks for `temperature 1.0, top_p 0.95` for agentic work; on a
shared server, prefer a non-greedy default and let clients that want greedy ask for it.

Images: see [Images](IMAGES.md).
