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

## Verification (all four probes WITNESSED 2026-08-21 against a loopback
## deployment of these exact configs — SELECTOR :1245 / RELAY :1246 stood in
## for :1235/:1236, leaves were the real 1301/1302/1401; re-witness on .68
## after the switch)

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

Witnessed outputs from the 2026-08-21 loopback run (leaves live, engines
built at `4e6fd3bd`):

```
query_version → {"id":"pv","action":"query_version","git_hash":"4e6fd3bd…","version":"1.17.2"}
query_models  → {"id":"pm","models":[{"label":"main","healthy":true},{"label":"alt","healthy":true}]}
analyze, no model → {"id":"pa","error":"missing 'model' field for SELECTOR routing","field":"model"}
analyze model=main / model=alt → moveInfos from the respective branch
12 distinct analyzes under "main" → relay dispatch events 10× :1301, 6× :1302, 0 fallbacks
```

## NN disk cache pairing (the point of the exercise)

The file locking this rests on is in this repo's KataGo as of commit
`f9be19cc`. Give **1301 and 1302
the same nnCache directory** in their katago configs — same model, same
host, shared page cache; concurrent attach is shared-locked, dumps are
exclusive-locked. Give **1401 its own directory**: a different model cannot
corrupt a shared cache (keys include model identity) but would bloat every
container with entries the other model's leaves load and never use.

## Shared-cache bring-up: probe the directory, then run the suite

Do these two steps **before** 1301 and 1302 are pointed at a shared
directory, and repeat them whenever that directory moves. Both commands are
safe to run against a live-but-idle directory and clean up after themselves.

**Run them on the machine the ENGINES run on, against the path the engines
use.** How that path is provided — local disk, sshfs, SMB, NFS — is the
storage operator's business and changes nothing here. Two transport-agnostic
facts are all this section relies on: (1) a lock verdict is only meaningful
from the engines' own machine and path, which is why the probe exists as a
command rather than a mount-options checklist; (2) on any network
filesystem, locks taken by the engines do not necessarily bind processes
running directly on the storage server — so while engines are live, nothing
on the storage side should write those files, and a direct-attach reader
there may see a torn tail mid-dump (the reader survives by discarding;
harmless to the store, but the view is truncated). Passive watching
(`ls -l`, sizes, mtimes) on the storage side is always fine.

A probe + suite run directly on the storage directory (done 2026-08-21:
SUPPORTED, 16 legs green) is the healthy-storage baseline; the verdict that
gates sharing is the one from the engines' side.

### Step 1 — does file locking actually EXCLUDE there?

```sh
mkdir -p ~/nncache                  # or mount it; KataGo will not create it
./cpp/build/katago lockfsprobe ~/nncache; echo "exit=$?"
```

Exit **0** = `LOCKFS VERDICT: SUPPORTED`, safe to share. Exit **1** =
`UNSUPPORTED`, **do not** point 1301 and 1302 at that directory — give each
leaf its own instead. Exit **2** = the probe could not run (directory
missing, not a directory), which is not a verdict.

This is the step that cannot be skipped or replaced by reasoning. The probe
forks a second process and makes the two contend for a real
`NNCacheFileLock` in the target directory, because a single process calling
`flock()` and getting `0` back has learned nothing — that is exactly what a
no-op `flock` returns too. sshfs is FUSE, and FUSE lock semantics depend on
the mount options in use (`-o nolock`-style behaviour, or a server that does
not support it), so the answer for `~/nncache` is an empirical one and this
is how it is obtained. A `SUPPORTED` verdict is printed only after one
process was made to *wait* for the other and then observed to acquire.

### Step 2 — do two real engines actually share it?

```sh
python3 cpp/tests/e2e/sharedcache_e2e_witness.py --cache-dir ~/nncache
echo "exit=$?"
```

Sixteen legs across three scenarios, driving real `katago analysis`
processes over the analysis protocol; roughly four seconds against a local
disk. Prints `PASS`/`FAIL` per leg with the numbers it read. Exit **0** =
all green, **1** = a leg failed, **2** = it could not run (and it runs
`lockfsprobe` itself first, so a directory that cannot lock is refused in
plain text with nothing written and nothing started).

- **SC1** one process evaluates and dumps; a later process attaches and is
  served entirely by the first one's keys — witnessed by three independent
  numbers, `entriesInLevelZero`, `NN rows: 0`, and a second dump with
  nothing to add.
- **SC2** the same, with **both engines alive at once**, which is the
  1301/1302 shape.
- **SC3** one engine attaches over and over *while* the other dumps; every
  observation must show the whole pre-dump or whole post-dump state, never
  anything in between and never a torn tail.

Useful flags: `--model` (defaults to `/home/bork/kg/14.gz`), `--config` (an
analysis config template holding `@CACHE_DIR@`), `--keep` to leave this
run's cache files behind for inspection, `--only sc3` to run one scenario.
`--cache-dir` may also be given as `$KATAGO_SHARED_CACHE_DIR`. Run it
against a local directory first: that establishes what a healthy result
looks like on this host, so a difference on the mount is attributable to
the mount.

## Rollback

Stop selector and relay on .68; restart the previous single-backend proxy
on :1235. The leaves need no changes — nothing in this topology writes
state into them beyond the shared cache directory pairing above.
