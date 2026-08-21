# KataProxy GPU topology — two same-model leaves + one alt-model leaf

Runbook for switching the running proxy at `ws://192.168.122.68:1235` from a
single backend to this topology, using ONLY configuration — no KataProxy code
or message-translation changes (ledger row 1596).

```
SPA ──ws──> SELECTOR  .68:1235          routes by wire `model` label
              ├─ "main" ──> RELAY  .68:1236 (loopback)
              │               ├──> LEAF .1:1301  ─ model A ┐ shared model,
              │               └──> LEAF .1:1302  ─ model A ┘ hash-ring balanced
              └─ "alt"  ─────────> LEAF .1:1401  ─ model B
```

## Why this shape (and not a flat ring of three)

- **RELAY's invariant is interchangeability**: its hash ring assumes any
  upstream can serve any query. That holds for 1301/1302 (same model) and is
  FALSE for 1401, so 1401 must never appear in `UPSTREAM_URLS`.
- **SELECTOR's invariant is distinctness**: labelled-dictionary dispatch, no
  LoadMetric, no fallback — "a query for model X cannot be served by model Y"
  (router.py:1601ff). Model separation is enforced by construction: no code
  path exists from the `alt` label to the ring.
- The operator's guess `{{1301,1302},{1401}}` is exactly this.

## Config files

| file | role | listens | upstreams |
|---|---|---|---|
| `relay.env` | RELAY | 127.0.0.1:1236 | ws://192.168.122.1:1301, :1302 |
| `selector.env` | SELECTOR | 0.0.0.0:1235 | main→127.0.0.1:1236, alt→192.168.122.1:1401 |

## Bring-up order

Leaves first, then relay, then selector — each tier probes its upstreams at
connect time and reconnects with backoff if they flap, but a clean first
connect gives clean logs.

1. **Leaves (operator, on 192.168.122.1)**: KataProxy LEAF processes wrapping
   katago on ports 1301, 1302 (same `KATAGO_MODEL`) and 1401 (different
   `KATAGO_MODEL`). Per leaf: `PROXY_ROLE=LEAF`, `PROXY_HOST=0.0.0.0` (the
   .68 VM must reach them), `PROXY_PORT=<port>`, `KATAGO_PATH`, `KATAGO_CFG`,
   `KATAGO_MODEL`.
2. **Relay (on .68, kataproxy checkout)**:
   `set -a; source relay.env; set +a; python ./proxy_server.py`
3. **Stop the currently running proxy on .68:1235**, then **selector**:
   `set -a; source selector.env; set +a; python ./proxy_server.py`

## Verification (wire examples UNWITNESSED — no websockets client was
## available from the authoring host; witness them here at first bring-up)

Against `ws://192.168.122.68:1235`, e.g. with `websocat`:

1. `{"id":"p1","action":"query_version"}` → one version response. The
   SELECTOR broadcasts this to every healthy upstream and the RELAY
   re-broadcasts to both leaves — this heartbeat fanout is load-bearing:
   leaves that never see it fire their KeepAlive watchdog mid-analysis
   (SELECTOR watchdog postmortem, cited in router.py).
2. `{"id":"p2","action":"query_models"}` →
   `{"id":"p2","models":[{"label":"main","healthy":true},{"label":"alt","healthy":true}]}`
   (synthesized by the SELECTOR from its label set; the SPA's model dropdown
   reads exactly this).
3. An analyze query **must carry `"model":"main"` or `"model":"alt"`** — a
   missing or unknown label returns a structured error naming the available
   labels. This is by design (fail loudly), not a bug. If the SPA today
   sends no `model` field, set its model from the dropdown once before
   analyzing.
4. Same-model balancing: submit several DIFFERENT positions under `main` and
   watch both 1301's and 1302's logs receive dispatches; identical repeat
   queries deterministically prefer one leaf (consistent hashing) — that is
   correct, not a stuck balancer.

## NN disk cache pairing (the point of the exercise)

Once the file-locking work lands in this repo's KataGo: give **1301 and 1302
the same nnCache directory** in their katago configs — same model, same
host, shared page cache; concurrent attach is shared-locked, dumps are
exclusive-locked. Give **1401 its own directory**: a different model cannot
corrupt a shared cache (keys include model identity) but would bloat every
container with entries the other model's leaves load and never use.

## Rollback

Stop selector and relay on .68; restart the previous single-backend proxy
on :1235. The leaves need no changes — nothing in this topology writes
state into them beyond the shared cache directory pairing above.
