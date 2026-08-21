# KataGo Parallel Analysis Engine

KataGo contains an engine that can be used to analyze large numbers of positions in parallel (entire games, or multiple games).
When properly configured and used with modern GPUs that can handle large batch sizes, this engine can be much faster than using
the GTP engine and `kata-analyze`, due to being able to take advantage of cross-position batching, and hopefully having a
nicer API. The analysis engine is primarily intended for people writing tools - for example, to run as the backend of an analysis
server or website.

This engine can be run via:

```./katago analysis -config CONFIG_FILE -model MODEL_FILE```

An example config file is provided in `cpp/configs/analysis_example.cfg`. Adjusting this config is recommended, for example
`nnCacheSizePowerOfTwo` based on how much RAM you have, and adjusting `numSearchThreadsPerAnalysisThread` (the number of MCTS threads operating simultaneously on the same position) and `numAnalysisThreads` (the number of positions that will be analyzed at the same time, *each* of which will use `numSearchThreadsPerAnalysisThread` many search threads).

See the [example analysis config](https://github.com/lightvector/KataGo/blob/master/cpp/configs/analysis_example.cfg#L60) for a fairly detailed discussion of how to tune these parameters.

## Hosting more than one model

The engine can load more than one neural net at once, and a query can say which one should analyze it:

```./katago analysis -config CONFIG_FILE -model MODEL_FILE -extra-model OTHER_MODEL_FILE```

`-extra-model` may be repeated for as many models as you want to host. Each hosted model gets its own
NN cache and its own pool of search bots, one per analysis thread, so a query is analyzed by a bot that
was built around that model rather than by swapping a net into a shared bot. Hosting N models therefore
costs roughly N times the NN cache and bot memory; it does not change how many positions are analyzed at
once, which is still `numAnalysisThreads`.

A query selects its model with the `model` field, whose value is the model's `internalName` exactly as
the `query_models` action reports it (`query_models` lists every hosted model). This is the model file's
own self-declared name; the engine does not invent an alias for it.

Two rules keep that name honest, and both refuse rather than guess:

  * **A query naming a model that is not loaded is an error, and is not analyzed.** The error message
    lists the names that are loaded. The engine will not fall back to the default model, because a
    response carries no record of which net produced it, so a silent fallback would be indistinguishable
    from the analysis you asked for.
  * **Two models that declare the same `internalName` are refused at startup**, with a message naming the
    name and both model files. This happens if you pass the same file twice, or two files that were built
    with the same name. There is no unambiguous answer to a query naming a name two models share, so the
    engine does not start rather than pick one.

A query with no `model` field behaves exactly as it always has.

`model` selects which model analyzes a *query*. Most action queries do not read it, so including `model`
on `query_models`, `clear_cache`, `terminate`, `terminate_all` or `query_version` is an error rather than
being ignored. The **cache actions** (`cache_attach`, `cache_detach`, `cache_dump`, `cache_stats`) do read
it - there it selects whose cache the action addresses - and they follow the same two rules above: an
unknown name is an error naming what is loaded, and no `model` field means the model started with
`-model`.

## Attributing what a query earns in the cache

A client may attach one or more named **cache contexts** to a model. A context is an opaque string the
client chooses: KataGo compares it for equality, checks it is usable as a filename component, and stores
it. It understands nothing else by it - there is no relation between contexts, no ordering and no
meaning attached to what a context stands for.

Contexts exist so that the new entries a session puts into a model's cache can be told apart. The cache
key names a position and nothing else, so with two contexts attached the engine cannot tell which of them
a query served. A query says so itself, with the optional `cacheContext` field:

  * A `cacheContext` naming a context attached to the model the query selected attributes that query's
    new cache entries to it.
  * **A `cacheContext` that is not attached to that model is an error, and the query is not analyzed.**
    The error message names it and lists the contexts that are attached. The engine will not fall back to
    another context, because that would file this session's work under the wrong one and nothing in the
    response would say it had happened.
  * With exactly **one** context attached to the model, a query with no `cacheContext` is attributed to
    it: with one attached, everything the session earns belongs to it.
  * With any other number attached, a query with no `cacheContext` earns entries that are counted and
    reported as **unattributed**. They are never assigned to whichever context happens to be first.

Contexts are per model. Each hosted model has its own cache and its own set of attached contexts, and a
name attached to one model says nothing about another - which is why `cacheContext` is resolved against
the model the same query selected with `model`.

A query with no `cacheContext` field behaves exactly as it always has, and so does an engine with no
context attached to anything, which is every engine that was started without `nnCacheDir`. Contexts are
attached with the `cache_attach` action; see "Persisting a model's cache across sessions" below.

`cacheContext` attributes a *query*'s new cache entries; no action query reads it, so including
`cacheContext` on an action query is an error rather than being ignored. The cache actions name the
context they act on in their own `context` field, which answers a different question - which body of
persisted content this action is *about*, rather than which one a query's earnings belong to.

## Persisting a model's cache across sessions

Normally the neural net cache lives and dies with the engine process: everything it computed is gone
when you exit. If you study the same positions over many sessions, you can instead have KataGo keep
them on disk, so a later session gets them back without paying for the evaluations again.

**This is off unless you configure a directory for it.** Add to your analysis config:

```
nnCacheDir = /some/existing/directory
```

KataGo owns every byte under that directory and will not create it - if the path is not an existing
directory the engine refuses to start, naming the path, so that a typo does not put your persisted cache
somewhere you will never look for it. Setting `nnCacheDir` while the cache itself is disabled
(`nnCacheSizePowerOfTwo` negative) is likewise refused at startup.

With `nnCacheDir` set, every hosted model's cache gains a **level 0**: an immutable, pre-warmed body of
evaluations that lookups consult before the ordinary cache. That is what a session attaches and detaches.
Without `nnCacheDir` the cache is exactly what it always was, and every cache action below is refused
with a message naming the config key.

### Contexts and files

Content on disk is grouped by **context** - the same opaque client-chosen name described under
"Attributing what a query earns in the cache". A context name must be usable as a filename component:
ASCII letters, digits, `.`, `_` and `-`, at most 128 characters, and not `.` or `..`. Anything else is
refused, never rewritten into something acceptable.

Each context has up to two files under `nnCacheDir`:

  * `<context>.nncounts` - how many times each position was retrieved, and in how many dumps it earned a
    retrieval. It is shared by every model, because a position's hash does not depend on which net
    evaluated it.
  * `<context>.<model>.nnevals` - the evaluations themselves, for one model. The model's `internalName`
    is both in the filename and inside the file's header, so **one file only ever holds one net's
    evaluations**. This is what keeps two nets' evaluations of the same position from ever being
    confused for each other: a position's cache key is the same under every net, so model identity lives
    in the file rather than in the key.

Both files are append-only logs with per-block checksums. If the engine is killed mid-write, the next
session reads everything up to the last complete block and reports how many bytes it discarded; the
partial tail is repaired by the next write, not by the read.

A missing file is not an error. Attaching a context nothing has ever written gives you an empty level 0,
which is a perfectly good thing to analyze against and then dump.

### The session shape

```
cache_attach   -> analyze, analyze, analyze ... -> cache_dump -> cache_detach
```

`cache_attach` and `cache_detach` are **session-boundary actions and are refused while any analysis
request is open**, naming how many are open. This is not a caution you can override: a cache lookup walks
the list of attached content without taking a lock, so changing that list under a live request would read
freed memory. Since attaching and detaching is what you do between sessions, when nothing is in flight,
this costs a normal client nothing. `cache_dump` and `cache_stats` have no such restriction.

**"Any request" means any request in the engine, not any request on this model.** The engine keeps one
set of open requests across every hosted model (see "Hosting more than one model"), so an open query
against model B refuses a `cache_attach` on model A. That is stricter than it needs to be - the two
models' caches are separate structures - and it is deliberately the strict direction, but if you host
several models and drive them from independent client threads, expect attach and detach to need a moment
when the *whole engine* is quiet, not just the model you are attaching to.

`cache_dump` is the **only** action that writes. Nothing is persisted automatically, on a timer, or at
exit - if you do not dump, nothing you computed reaches disk.

A context's name stays registered with a model for the life of the process once you have attached it, even
after you detach. Re-attaching the same context later is fine and reuses that registration; attaching a
context that is *currently* attached is an error.

### Reading an error from a cache action

Errors use the engine's usual shape - an object with `id`, `field` and `error`. Two kinds of
`field` value appear, and the difference is only where the refusal came from:

  * A field name (`"context"`, `"what"`, `"model"`, `"level0"`, ...) means the request itself could
    not be read. Nothing was attempted. A context name outside the legal alphabet comes back this
    way, under `"context"`, alongside the missing, empty and wrong-typed cases.
  * `"action"` means the request was well-formed and the engine refused to carry it out - the
    context is not attached, the detach would lose undumped work, a request is open, the file on
    disk is not readable. The `error` text says which. Nothing was left half-done: an attach that
    fails partway takes back whatever it had already attached before returning the error.

### Unknown fields are errors here

Every field of a cache action decides which bytes on disk are read or written. So, unlike an analysis
query - where an unrecognized top-level field produces a warning and the query is analyzed anyway, and
where that warning can be switched off with `warnUnusedFields=false` - **an unrecognized field on a cache
action is an error and the action is not performed**. The error names the field and lists the fields that
action accepts. There is no config key that turns this into a warning.

## Example Code

For example code demonstrating how to invoke the analysis engine from Python, see [here](https://github.com/lightvector/KataGo/blob/master/python/query_analysis_engine_example.py)!

## Protocol

The engine accepts queries on stdin, and output results on stdout. Every query and every result should be a single line.
The protocol is entirely asynchronous - new requests on stdin can be accepted at any time, and results will appear on stdout
whenever those analyses finish, and possibly in a different order than the requests were provided. As described below, each query
may specify *multiple* positions to be analyzed and therefore may generate *multiple* results.

If stdin is closed, then the engine will finish the analysis of all queued queries before exiting, unless `-quit-without-waiting` was
provided on the initial command line, in which case it will attempt to stop all threads and still exit cleanly but without
necessarily finishing the analysis of whatever queries are open at the time.

### Queries

Each query line written to stdin should be a JSON dictionary with certain fields. Note again that every query must be a *single line* - multi-line JSON queries are NOT supported. An example query would be:

```json
{"id":"foo","initialStones":[["B","Q4"],["B","C4"]],"moves":[["W","P5"],["B","P6"]],"rules":"tromp-taylor","komi":7.5,"boardXSize":19,"boardYSize":19,"analyzeTurns":[0,1,2]}
```

<details>
<summary>
See formatted query for readability (but note that this is not valid input for KataGo, since it spans multiple lines).
</summary>

```json
{
    "id": "foo",
    "initialStones": [
        ["B", "Q4"],
        ["B", "C4"]
    ],
    "moves": [
        ["W", "P5"],
        ["B", "P6"]
    ],
    "rules": "tromp-taylor",
    "komi": 7.5,
    "boardXSize": 19,
    "boardYSize": 19,
    "analyzeTurns": [0, 1, 2]
}
```
</details>

This example query specifies a 2-stone handicap game record with certain properties, and requests analysis of turns 0,1,2 of the game, which should produce three results.

Explanation of fields (including some optional fields not present in the above query):

   * `id (string)`: Required. An arbitrary string identifier for the query.
   * `moves (list of [player string, location string] tuples)`: Required. The moves that were played in the game, in the order they were played.
     * `player string` should be `"B"` or `"W"`.
     * `location` should a string like `"C4"` the same as in the [GTP protocol](http://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html#SECTION000311000000000000000). KataGo also supports extended column coordinates locations beyond `"Z"`, such as `"AA"`, `"AB"`, `"AC"`, ... Alternatively one can also specify strings like `"(0,13)"` that explicitly give the integer X and Y coordinates.
     * Leave this array empty if you have an initial position with no move history (do not make up an arbitrary or "fake" order of moves).
   * `initialStones (list of [player string, location string] tuples)`: Optional. Specifies stones already on the board at the start of the game. For example, these could be handicap stones. Or, you could use this to specify a midgame position or whole-board tsumego that does not have a move history.
     * If you know the real game moves that reached a position, using `moves` is usually preferable to specifying all the stones here while leaving `moves` as an empty array, since using `moves` ensures correct ko/superko handling, and the neural net may also take into account the move history in its future predictions.
   * `initialPlayer (player string)`: Optional. Specifies the player to use for analyzing the first turn (turn 0) of the game, which can be useful if `moves` is an empty list.
   * `rules (string or JSON)`: Required. Specify the rules for the game using either a shorthand string or a full JSON object.
     * See the documentation of `kata-get-rules` and `kata-set-rules` in [GTP Extensions](./GTP_Extensions.md) for a description of supported rules.
     * Some older neural net versions of KataGo do not support some rules options. If this is the case, then a warning will be issued and the rules will
       automatically be converted to the nearest rules that the neural net does support.
   * `komi (integer or half-integer)`: Optional but HIGHLY recommended. Specify the komi for the game. If not specified, KataGo will guess a default value, generally 7.5 for area scoring, but 6.5 if using territory scoring, and 7.0 if area scoring with a button. Values of komi outside of [-400,400] are not supported.
   * `whiteHandicapBonus (0|N|N-1)`: Optional. See `kata-get-rules` in [GTP Extensions](./GTP_Extensions.md) for what these mean. Can be used to override the handling of handicap bonus, taking precedence over `rules`. E.g. if you want `chinese` rules but with different compensation for handicap stones than Chinese rules normally use. You could also always specify this as 0 and do any adjustment you like on your own, by reporting an appropriate `komi`.
   * `boardXSize (integer)`: Required. The width of the board. Sizes > 19 are NOT supported unless KataGo has been compiled to support them (cpp/game/board.h, MAX_LEN = 19). KataGo's official neural nets have also not been trained for larger boards, but should work fine for mildly larger sizes (21,23,25).
   * `boardYSize (integer)`: Required. The height of the board. Sizes > 19 are NOT supported unless KataGo has been compiled to support them (cpp/game/board.h, MAX_LEN = 19). KataGo's official neural nets have also not been trained for larger boards, but should work fine for mildly larger sizes (21,23,25).
   * `analyzeTurns (list of integers)`: Optional. Which turns of the game to analyze. 0 is the initial position, 1 is the position after `moves[0]`, 2 is the position after `moves[1]`, etc. If this field is not specified, defaults to analyzing only the last turn, which is the position after all specified `moves` are made.
   * `maxVisits (integer)`: Optional. The maximum number of visits to use. If not specified, defaults to the value in the analysis config file. If specified, overrides it.
   * `model (string)`: Optional. Which of the loaded models should analyze this query, given as the `internalName` that the `query_models` action reports for it. If not specified, the query is analyzed by the model given by `-model` on the command line, which is what the engine has always done and is the only model loaded unless `-extra-model` was also given. A name that is not a loaded model, or that names the human SL model (which participates in searches but is not independently searchable), is an ERROR for that query and nothing is analyzed - the engine will not quietly substitute a different net. See "Hosting more than one model" below.
   * `cacheContext (string)`: Optional. Which attached cache context this query's new cache entries are earned by, resolved against the model this query selects. A context that is not attached to that model is an ERROR for that query and nothing is analyzed - the engine will not attribute the work to some other context. If not specified, the query is attributed to the sole attached context when exactly one is attached, and is otherwise counted as unattributed. See "Attributing what a query earns in the cache" below.
   * `rootPolicyTemperature (float)`: Optional. Set this to a value > 1 to make KataGo do a wider search.
   * `rootFpuReductionMax (float)`: Optional. Set this to 0 to make KataGo more willing to try a variety of moves.
   * `analysisPVLen (integer)`: Optional. The maximum length of the PV to send for each move (not including the first move).
   * `includeOwnership (boolean)`: Optional. If true, report ownership prediction as a result. Will double memory usage and reduce performance slightly.
   * `includeOwnershipStdev (boolean)`: Optional. If true, report standard deviation of ownership predictions across the search as well.
   * `includeMovesOwnership (boolean)`: Optional. If true, report ownership prediction for every individual move too.
   * `includeMovesOwnershipStdev (boolean)`: Optional. If true, report stdev of ownership prediction for every individual move too.
   * `includePolicy (boolean)`: Optional. If true, report neural network raw policy as a result. Will not signficiantly affect performance.
   * `includePVVisits (boolean)`: Optional. If true, report the number of visits for each move in any reported pv.
   * `includeNoResultValue (boolean)`: Optional. If true, report the predicted no-result probability for each move.
   * `avoidMoves (list of dicts)`: Optional. Prohibit the search from exploring the specified moves for the specified player, until a certain number of ply deep in the search. Each dict must contain these fields:
      * `player` - the player to prohibit, `"B"` or `"W"`.
      * `moves` - an array of move locations to prohibit, such as `["C3","Q4","pass"]`
      * `untilDepth` - a positive integer, indicating the ply such that moves are prohibited before that ply.
      * Multiple dicts can specify different `untilDepth` for different sets of moves. The behavior is unspecified if a move is specified more than once with different `untilDepth`.
   * `allowMoves (list of dicts)`: Optional. Same as `avoidMoves` except prohibits all moves EXCEPT the moves specified. At most one dict may be specified per player (so the list may contain at most two dicts, one for `"B"` and one for `"W"`).
   * `overrideSettings (object)`: Optional. Specify any number of `"paramName":value` entries in this object to override those params from command line `CONFIG_FILE` for this query. Most search parameters can be overriden: `cpuctExploration`, `winLossUtilityFactor`, etc. Some notable parameters include:
      * `playoutDoublingAdvantage (float)`. A value of PDA from -3 to 3 will adjust KataGo's evaluation to assume that the opponent is NOT of equal strength/compte, but rather that the current player has 2^(PDA) times as many playouts as the opponent. Dynamic versions of this are used to significant effect in handicap games in GTP mode, see [GTP example config](../cpp/configs/gtp_example.cfg).
      * `wideRootNoise (float)`. See documentation for this parameter in [the example config](../cpp/configs/analysis_example.cfg)
      * `ignorePreRootHistory (boolean)`. Whether to ignore pre-root history during analysis.
      * `antiMirror (boolean)`. Whether to enable anti-mirror play during analysis. Off by default. Will probably result in biased and nonsensical winrates and other analysis values, but moves may detect and crudely respond to mirror play.
      * `rootNumSymmetriesToSample (int from 1 to 8)`. How many of the 8 possible random symmetries to evaluate the neural net with and average. Defaults to 1, but if you set this to 2, or 8 you might get a slightly higher-quality policy at the root due to noise reduction.
      * `humanSLProfile (string)`. Set the human-like play that KataGo should imitate. Requires that a human SL model like `b18c384nbt-humanv0.bin.gz` is being used, typically via the command line parameter `-human-model`. Available profiles include:
           * `preaz_20k` through `preaz_9d`: Imitate human players of the given rank. (based on 2016 pre-AlphaZero opening style).
           * `rank_20k` through `rank_9d`: Imitate human players of the given rank (modern opening style).
           * `preaz_{BR}_{WR}` or `rank_{BR}_{WR}`: Same, but predict how black with the rank BR and white with the rank WR would play against each other, *knowing* that the other player is stronger/weaker than them. Warning: for rank differences > 9 ranks, or drastically mis-matched to the handicap used in the game, this may be out of distribution due to lack of training data and the model might not behave well! Experiment with care.
           * `proyear_1800` through `proyear_2023`: Imitate pro and strong insei moves based on historical game records from the specified year and surrounding years.
           * See also section below, "Human SL Analysis Guide" for various other parameters that are interesting to set in conjunction with this.
   * `reportDuringSearchEvery (float)`: Optional. Specify a number of seconds such that while this position is being searched, KataGo will report the partial analysis every that many seconds.
   * `priority (int)`: Optional. Analysis threads will prefer handling queries with the highest priority unless already started on another task, breaking ties in favor of earlier queries. If not specified, defaults to 0.
   * `priorities (list of integers)`: Optional. When using analyzeTurns, you can use this instead of `priority` if you want a different priority per turn. Must be of same length as `analyzeTurns`, `priorities[0]` is the priority for `analyzeTurns[0]`, `priorities[1]` is the priority for `analyzeTurns[1]`, etc.


### Responses

Upon an error or a warning, responses will have one of the following formats:
```
# General error
{"error":"ERROR_MESSAGE"}
# Parsing error for a particular query field
{"error":"ERROR_MESSAGE","field":"name of the query field","id":"The id string for the query with the error"}
# Parsing warning for a particular query field
{"warning":"WARNING_MESSAGE","field":"name of the query field","id":"The id string for the query with the error"}
```
In the case of a warning, the query will still proceed to generate analysis responses.

An example successful analysis response might be:
```json
{"id":"foo","isDuringSearch":false,"moveInfos":[{"lcb":0.8740855166489953,"move":"Q5","order":0,"prior":0.8934692740440369,"pv":["Q5","R5","Q6","P4","O5","O4","R6","S5","N4","N5","N3"],"scoreLead":8.18535151076558,"scoreMean":8.18535151076558,"scoreSelfplay":10.414442461570038,"scoreStdev":23.987067985850913,"utility":0.7509536097709347,"utilityLcb":0.7717092488727239,"visits":495,"edgeVisits":495,"winrate":0.8666727883983563},{"lcb":1.936558574438095,"move":"D4","order":1,"prior":0.021620146930217743,"pv":["D4","Q5"],"scoreLead":12.300520420074463,"scoreMean":12.300520420074463,"scoreSelfplay":15.386500358581543,"scoreStdev":24.661467510313432,"utility":0.9287495791972984,"utilityLcb":2.8000000000000003,"visits":2,"edgeVisits":2,"winrate":0.9365585744380951},{"lcb":1.9393062554299831,"move":"Q16","order":2,"prior":0.006689758971333504,"pv":["Q16"],"scoreLead":12.97426986694336,"scoreMean":12.97426986694336,"scoreSelfplay":16.423904418945313,"scoreStdev":25.34494674587838,"utility":0.9410896213959669,"utilityLcb":2.8000000000000003,"visits":1,"edgeVisits":1,"winrate":0.9393062554299831},{"lcb":1.9348860532045364,"move":"D16","order":3,"prior":0.0064553022384643555,"pv":["D16"],"scoreLead":12.066888809204102,"scoreMean":12.066888809204102,"scoreSelfplay":15.591397285461426,"scoreStdev":25.65390196745236,"utility":0.9256971928661066,"utilityLcb":2.8000000000000003,"visits":1,"edgeVisits":1,"winrate":0.9348860532045364}],"rootInfo":{"currentPlayer":"B","lcb":0.8672585456293346,"scoreLead":8.219540952281882,"scoreSelfplay":10.456476293719811,"scoreStdev":23.99829921716391,"symHash":"1D25038E8FC8C26C456B8DF2DBF70C02","thisHash":"F8FAEDA0E0C89DDC5AA5CCBB5E7B859D","utility":0.7524437705003542,"visits":500,"winrate":0.8672585456293346},"turnNumber":2}
```
<details>
<summary>
See formatted response.
</summary>

```json
{
    "id": "foo",
    "isDuringSearch": false,
    "moveInfos": [{
        "lcb": 0.8740855166489953,
        "move": "Q5",
        "order": 0,
        "prior": 0.8934692740440369,
        "pv": ["Q5", "R5", "Q6", "P4", "O5", "O4", "R6", "S5", "N4", "N5", "N3"],
        "scoreLead": 8.18535151076558,
        "scoreMean": 8.18535151076558,
        "scoreSelfplay": 10.414442461570038,
        "scoreStdev": 23.987067985850913,
        "utility": 0.7509536097709347,
        "utilityLcb": 0.7717092488727239,
        "visits": 495,
        "edgeVisits": 495,
        "winrate": 0.8666727883983563
    }, {
        "lcb": 1.936558574438095,
        "move": "D4",
        "order": 1,
        "prior": 0.021620146930217743,
        "pv": ["D4", "Q5"],
        "scoreLead": 12.300520420074463,
        "scoreMean": 12.300520420074463,
        "scoreSelfplay": 15.386500358581543,
        "scoreStdev": 24.661467510313432,
        "utility": 0.9287495791972984,
        "utilityLcb": 2.8000000000000003,
        "visits": 2,
        "edgeVisits": 2,
        "winrate": 0.9365585744380951
    }, {
        "lcb": 1.9393062554299831,
        "move": "Q16",
        "order": 2,
        "prior": 0.006689758971333504,
        "pv": ["Q16"],
        "scoreLead": 12.97426986694336,
        "scoreMean": 12.97426986694336,
        "scoreSelfplay": 16.423904418945313,
        "scoreStdev": 25.34494674587838,
        "utility": 0.9410896213959669,
        "utilityLcb": 2.8000000000000003,
        "visits": 1,
        "edgeVisits": 1,
        "winrate": 0.9393062554299831
    }, {
        "lcb": 1.9348860532045364,
        "move": "D16",
        "order": 3,
        "prior": 0.0064553022384643555,
        "pv": ["D16"],
        "scoreLead": 12.066888809204102,
        "scoreMean": 12.066888809204102,
        "scoreSelfplay": 15.591397285461426,
        "scoreStdev": 25.65390196745236,
        "utility": 0.9256971928661066,
        "utilityLcb": 2.8000000000000003,
        "visits": 1,
        "edgeVisits": 1,
        "winrate": 0.9348860532045364
    }],
    "rootInfo": {
        "currentPlayer": "B",
        "lcb": 0.8672585456293346,
        "scoreLead": 8.219540952281882,
        "scoreSelfplay": 10.456476293719811,
        "scoreStdev": 23.99829921716391,
        "symHash":"1D25038E8FC8C26C456B8DF2DBF70C02",
        "thisHash":"F8FAEDA0E0C89DDC5AA5CCBB5E7B859D",
        "utility": 0.7524437705003542,
        "visits": 500,
        "winrate": 0.8672585456293346
    },
    "turnNumber": 2
}
```
</details>


**All values will be from the perspective of `reportAnalysisWinratesAs` as specified in the analysis config file.**

Consumers of this data should attempt to be robust to possible addition of both new top-level fields in the future, as well as additions to fields in `moveInfos` or `rootInfo`.

The various "human" fields are available if -human-model is provided and humanSLProfile is set.

Current fields are:

   * `id`: The same id string that was provided on the query.
   * `isDuringSearch`: Normally false. If `reportDuringSearchEvery` is provided, then will be true on the reports during the middle of the search before the search is complete. Every position searched will still always conclude with exactly one final response when the search is completed where this field is false.
   * `turnNumber`: The turn number being analyzed.
   * `moveInfos`: A list of JSON dictionaries, one per move that KataGo considered, with fields indicating the results of analysis. Current fields are:
      * `move` - The move being analyzed.
      * `visits` - The number of visits that the child node received.
      * `edgeVisits` - The number of visits that the root node "wants" to invest in the move, due to thinking it's a plausible or search-worthy move. Might differ from `visits` due to human SL weightless exploration, or graph search transpositions.
      * `winrate` - The winrate of the move, as a float in [0,1].
      * `scoreMean` - Same as scoreLead. "Mean" is a slight misnomer, but this field exists to preserve compatibility with existing tools.
      * `scoreStdev` - The predicted standard deviation of the final score of the game after this move, in points. (NOTE: due to the mechanics of MCTS, this value will be **significantly biased high** currently, although it can still be informative as *relative* indicator).
      * `scoreLead` - The predicted average number of points that the current side is leading by (with this many points fewer, it would be an even game).
      * `scoreSelfplay` - The predicted average value of the final score of the game after this move during selfplay, in points. (NOTE: users should usually prefer scoreLead, since scoreSelfplay may be biased by the fact that KataGo isn't perfectly score-maximizing).
      * `prior` - The policy prior of the move, as a float in [0,1].
      * `noResultValue` - The predicted probability that the game ends in a no-result (e.g. triple ko/long cycle), as a float in [0,1]. Only present if `includeNoResultValue` was true. Relevant for rulesets like Japanese rules without superko.
      * `humanPrior` - The human policy for the move, as a float in [0,1], if available.
      * `utility` - The utility of the move, combining both winrate and score, as a float in [-C,C] where C is the maximum possible utility. The maximum winrate utility can be set by `winLossUtilityFactor` in the config, while the maximum score utility is the sum of `staticScoreUtilityFactor` and `dynamicScoreUtilityFactor`.
      * `lcb` - The [LCB](https://github.com/leela-zero/leela-zero/issues/2282) of the move's winrate. Has the same units as winrate, but might lie outside of [0,1] since the current implementation doesn't strictly account for the 0-1 bounds.
      * `utilityLcb` - The LCB of the move's utility.
      * `weight` - The total weight of the visits that the child node received. The average weight of visits may be lower when less certain, and larger when more certain.
      * `edgeWeight` - The total weight of the visits the parent wants to invest into the move. The average weight of visits may be lower when less certain, and larger when more certain.
      * `order` - KataGo's ordinal ranking of the move. 0 is the best, 1 is the next best, and so on.
      * `playSelectionValue` - The value used to compute `order`. KataGo chooses the move with the maximum playSelectionValue, which is based on a combination of winrate, score, and other properties. When playing with randomization (i.e. in GTP, rather than the Analysis Engine), KataGo chooses moves proportional to this value raised to a constant depending on the temperature.
      * `isSymmetryOf` - Another legal move. Possibly present if KataGo is configured to avoid searching some moves due to symmetry (`rootSymmetryPruning=true`). If present, this move was not actually searched, and all of its stats and PV are copied symmetrically from that other move.
      * `pv` - The principal variation ("PV") following this move. May be of variable length or even empty.
      * `pvVisits` - The number of visits used to explore the position resulting from each move in `pv`. Exists only if `includePVVisits` is true.
      * `pvEdgeVisits` - The number of visits used to explore each move in `pv`. Exists only if `includePVVisits` is true. Differs from pvVisits when doing graph search and multiple move sequences lead to the same position - pvVisits will count the total number of visits for the position at that point in the PV, pvEdgeVisits will count only the visits reaching the position using the move in the PV from the preceding position.
      * `ownership` - If `includeMovesOwnership` was true, then this field will be included. It is a JSON array of length `boardYSize * boardXSize` with values from -1 to 1 indicating the predicted ownership after this move. Values are in row-major order, starting at the top-left of the board (e.g. A19) and going to the bottom right (e.g. T1).
      * `ownershipStdev` - If `includeMovesOwnershipStdev` was true, then this field will be included. It is a JSON array of length `boardYSize * boardXSize` with values from 0 to 1 indicating the per-location standard deviation of predicted ownership in the search tree after this move. Values are in row-major order, starting at the top-left of the board (e.g. A19) and going to the bottom right (e.g. T1).
   * `rootInfo`: A JSON dictionary with fields containing overall statistics for the requested turn itself calculated in the same way as they would be for the next moves. Current fields are: `winrate`, `scoreLead`, `scoreSelfplay`, `utility`, `visits`. And additional fields:
      * `thisHash` - A string that will with extremely high probability be unique for each distinct (board position, player to move, simple ko ban) combination.
      * `symHash` - Like `thisHash` except the string will be the same between positions that are symmetrically equivalent. Does NOT necessarily take into account superko.
      * `currentPlayer` - The current player whose possible move choices are being analyzed, `"B"` or `"W"`.
      * `rawWinrate` - The winrate prediction of the neural net by itself, without any search.
      * `rawLead` - The lead prediction of the neural net by itself, without any search.
      * `rawScoreSelfplay` - The selfplay score prediction of the neural net by itself, without any search.
      * `rawScoreSelfplayStdev` - The standard deviation of the final game score predicted by the net itself, without any search.
      * `rawNoResultProb` - The raw predicted probability of a no result game in Japanese-like rules.
      * `rawStWrError` - The short-term uncertainty the raw neural net believed there would be in the winrate of the position, prior to searching it.
      * `rawStScoreError` - The short-term uncertainty the raw neural net believed there would be in the score of the position, prior to searching it.
      * `rawVarTimeLeft` - The raw neural net's guess of "how long of a meaningful game is left?", in no particular units. A large number when expected that it will be a long game before the winner becomes clear. A small number when the net believes the winner is already clear, or that the winner is unclear but will become clear soon.
      * `humanWinrate` - Same as `rawWinrate` but using the human model, if available.
      * `humanScoreMean` - Same as `rawScoreSelfplay` but using the human model, if available.
      * `humanScoreStdev` - Same as `rawScoreSelfplayStdev` but using the human model, if available.
      * `humanStWrError` - The short-term uncertainty the raw neural net believes there will be in the winrate of the position as it gets played out by players of the configured profile, using the human model, if available.
      * `humanStScoreError` - The short-term uncertainty the raw neural net believes there will be in the score evaluation of the position as it gets played out by players of the configured profile, using the human model, if available.
      * Note that properties of the root like "winrate" and score will vary more smoothly and a bit more sluggishly than the corresponding property of the best move, since the rootInfo averages smoothly across all visits even while the top move may fluctuate rapidly. This may or may not be preferable over reporting the stats of the top move, depending on the purpose.
   * `ownership` - If `includeOwnership` was true, then this field will be included. It is a JSON array of length `boardYSize * boardXSize` with values from -1 to 1 indicating the predicted ownership. Values are in row-major order, starting at the top-left of the board (e.g. A19) and going to the bottom right (e.g. T1).
   * `ownershipStdev` - If `includeOwnershipStdev` was true, then this field will be included. It is a JSON array of length `boardYSize * boardXSize` with values from 0 to 1 indicating the per-location standard deviation of predicted ownership in the search tree. Values are in row-major order, starting at the top-left of the board (e.g. A19) and going to the bottom right (e.g. T1).
   * `policy` - If `includePolicy` was true, then this field will be included. It is a JSON array of length `boardYSize * boardXSize + 1` with positive values summing to 1 indicating the neural network's prediction of the best move before any search, and `-1` indicating illegal moves. Values are in row-major order, starting at the top-left of the board (e.g. A19) and going to the bottom right (e.g. T1). The last value in the array is the policy value for passing.
   * `humanPolicy` - If `includePolicy` was true, and a human model is available, then this field will be included. The format is the same as `policy`, but it reports the policy from the human model based on the configured `humanSLProfile`. See also section below, "Human SL Analysis Guide".


### Special Action Queries

Currently a few special action queries are supported that direct the analysis engine to do something other than enqueue a new position or set of positions for analysis.
A special action query is also sent as a JSON object, but with a different set of fields depending on the query.

#### query_version
Requests that KataGo report its current version. Required fields:

   * `id (string)`: Required. An arbitrary string identifier for this query.
   * `action (string)`: Required. Should be the string `query_version`.

Example:
```
{"id":"foo","action":"query_version"}
```

The response to this query is to echo back a json object with exactly the same data and fields of the query, but with two additional fields:

   * `version (string)`: A string indicating the most recent KataGo release version that this version is a descendant of, such as `1.6.1`.
   * `git_hash (string)`: The precise git hash this KataGo version was compiled from, or the string `<omitted>` if KataGo was compiled separately from its repo or without Git support.

Example:
```
{"action":"query_version","git_hash":"0b0c29750fd351a8364440a2c9c83dc50195c05b","id":"foo","version":"1.6.1"}
```

#### clear_cache
Requests that KataGo empty its neural net cache. Required fields:

   * `id (string)`: Required. An arbitrary string identifier for this query.
   * `action (string)`: Required. Should be the string `clear_cache`.

Example:
```
{"id":"foo","action":"clear_cache"}
```
The response to this query is to simply echo back a json object with exactly the same data and fields of the query. This response is sent after the cache is successfully cleared. If more than one model is hosted (see "Hosting more than one model"), every hosted model's cache is cleared, since the action means "drop everything cached" and a model whose cache it skipped would keep serving entries you asked to be rid of. If there are also any ongoing analysis queries at the time, those queries will of course be concurrently refilling the cache even as the response is being sent.

Explanation: KataGo uses a cache of neural net query results to skip querying the neural net when it encounters within its search tree a position whose stone configuration, player to move, ko status, komi, rules, and other relevant options are all identical a position it has seen before. For example, this may happen if the search trees for some queries overlap due to being on nearby moves of the same game, or it may happen even within a single analysis query if the search explores differing orders of moves that lead to the same positions (often, about 20% of search tree nodes hit the cache due transposing to order of moves, although it may be vastly higher or lower depending on the position and search depth). Reasons for wanting to clear the cache may include:

* Freeing up RAM usage - emptying the cache should release the memory used for the results in the cache, which is typically the largest memory usage in KataGo. Memory usage will of course rise again as the cache refills.

* Testing or studying the variability of KataGo's search results for a given number of visits. Analyzing a position again after a cache clear will give a "fresh" look on that position that better matches the variety of possible results KataGo may return, simliar to if the analysis engine were entirely restarted. Each query will re-randomize the symmetry of the neural net used for that query instead of using the cached result, giving a new and more varied opinion.

**What `clear_cache` does when a context is attached** (see "Persisting a model's cache across sessions").
Its behaviour is unchanged, and stated here because with persisted content in play "empty the cache" has
two possible readings and only one of them is right:

* **Attached level-0 content is NOT cleared.** It is the pre-warmed material a session was given, it
  cannot be rebuilt in the process, and discarding it would throw away the entire point of having
  attached it. Use `cache_detach` to give it back.
* **The ordinary cache - everything this session computed - is cleared**, on every hosted model, exactly
  as before.
* **Retrieval counts survive**, and so does the record of which context earned which entry. A
  `clear_cache` is not the end of a session, so a `cache_dump` after one still writes the retrievals that
  happened before it.
* Entries that were computed but not yet dumped are **gone**, and `clear_cache` does not warn about that
  the way `cache_detach` does. If you care about them, dump before you clear.


#### cache_attach

Attaches a context's persisted cache content to a model, for the session that is about to run. Requires
`nnCacheDir` (see "Persisting a model's cache across sessions"). Fields:

   * `id (string)`: Required. An arbitrary string identifier for this query.
   * `action (string)`: Required. Should be the string `cache_attach`.
   * `context (string)`: Required. The context to attach.
   * `model (string)`: Optional. Which model's cache to attach it to, by `internalName`. Defaults to the
     model started with `-model`.
   * `level0 (object)`: Optional. How much of the context goes into the frozen, pre-warmed level 0.
     Exactly one of:
       * `{"minLookups":N}` - every position the count log records at least `N` retrievals for. A position
         the count log has never mentioned is never admitted by this at any `N` above 0: its count is not
         zero, it is unknown.
       * `{"maxEntries":N}` - the `N` most-retrieved positions.
       * `{"maxBytes":B}` - the most-retrieved positions that fit in `B` bytes of memory. `B` counts the
         memory the entries will actually occupy, not the size of the file.
     Omitted means every persisted position. Two of them at once is an error: they would select two
     different sets and KataGo will not choose between them - send two attaches, or pick the one you mean.
   * `level1Fill`: Optional. What to do with the positions `level0` did not take. `false` (the default)
     leaves them on disk. `{"maxBytes":B}` admits the most-retrieved of them into the ordinary cache, up
     to `B` bytes. A bare `true` is an error: there is no byte budget KataGo could invent for you.
   * `foreignModelSources (array of string)`: Optional, and empty by default. Other loaded models whose
     `<context>.<model>.nnevals` files should also be attached, *after* this model's own, in the order
     you list them. See "Reading another model's evaluations" below. Listing a model twice, or listing
     the model you are attaching to, is an error.

Example:
```
{"id":"a1","action":"cache_attach","context":"card-5455","level0":{"minLookups":2},"level1Fill":{"maxBytes":268435456}}
```

The response echoes the query and adds:

   * `context`, `model`: what was attached, and to which model.
   * `entriesInLevelZero (integer)`: positions now in the frozen level 0, summed over all sources.
   * `levelZeroPayloadBytes`, `levelZeroStructureBytes (integer)`: the memory this attach is holding -
     the evaluations themselves, and the lookup structure over them. Both are returned by `cache_detach`.
   * `levelOneFilled (integer)`, `levelOneFilledBytes (integer)`: what the `level1Fill` admitted.
   * `sources (array of object)`: one entry per attached file, in the order they will be consulted, each
     with `model`, `entriesInLevelZero`, `entriesLeftOver`, `payloadBytes`, `structureBytes`,
     `entriesLevelOneAlreadyOwned` and `hitsTransferredToLevelOne`. The last two are what reconciling
     this file against the live cache did: an attach **shadows every arriving position the live cache
     already owns**, so a card re-attached after this session has already re-evaluated part of it cannot
     serve the superseded evaluation, and the retrievals those shadowed entries had accrued are handed to
     the counter that takes over rather than dropped. Both are `0` on an ordinary first attach; a large
     `entriesLevelOneAlreadyOwned` says how much of the file you just loaded is resident memory no lookup
     will reach while this attachment stands.
   * `containerTail`, `countLogTail (string)`: `"intact"`, or `"truncated"` if a previous run was killed
     mid-write. With `containerDiscardedTailBytes` / `countLogDiscardedTailBytes`, the bytes after the
     last complete block. Truncation is not an error - the intact prefix is what a crash left you, and is
     what you get - but a nonzero figure is worth noticing.
   * `buildMilliseconds (number)`: how long the read and the structure build took.

Errors, each of which leaves nothing attached:

   * The engine was started without `nnCacheDir`.
   * The context is already attached to this model.
   * `model`, or a name in `foreignModelSources`, is not a loaded model.
   * The context name is not a legal filename component.
   * A file exists under that name and is not one of KataGo's, is a version this build does not read, or
     names a different model than the one being attached for.

**Reading another model's evaluations.** By default a model reads only its own file, so two nets never
share an entry and there is no setting to get wrong. `foreignModelSources` is the opt-in for the case
where you have several nets, consider their evaluations interchangeable for your purposes, and want the
strongest one to be able to use what the weaker ones already computed. The list is the priority order:
the first source that holds a position answers for it, and this model's own file is always first, so its
own evaluations always win. Everything the session then computes is this model's and dumps to this
model's file - so over sessions the strong net's entries organically replace the weak ones'. Note what
you are asking for: an evaluation served this way was produced by a different net than the one analyzing.
That is why it is off by default, per-attach, and reported back per source.

#### cache_detach

Releases a context's attached content and gives the memory back. Fields:

   * `id (string)`: Required. An arbitrary string identifier for this query.
   * `action (string)`: Required. Should be the string `cache_detach`.
   * `context (string)`: Required. The context to detach.
   * `model (string)`: Optional. Defaults to the model started with `-model`.
   * `discardUndumped (boolean)`: Optional, default `false`. Whether you accept losing work that has not
     been dumped.

Example:
```
{"id":"a2","action":"cache_detach","context":"card-5455"}
```

**A detach with undumped work is refused**, and the refusal says how many earned positions are not
on disk. Silently throwing that work away would lose a session with nothing in the response to say so;
silently dumping it would make `cache_dump` no longer the only action that writes. So you either send
`cache_dump` first, or send the detach again with `"discardUndumped":true`, where the decision is visible
in your own log.

The refusal fires on either of two conditions, and both are exact:

  * **Undumped evaluations** - positions this context earned whose bytes are not in its file. The
    engine records, per position, whether its bytes reached disk.
  * **Undumped retrieval counts** - retrievals this context has served, by either level, that no
    `cache_dump` has written. The engine asks this **without consuming them**: the query reads the same
    counters against the same marks that a dump would advance, and moves none of them.

**What changed here, if you read the previous version of this section.** The counts condition used to be
a proxy - whether the engine had *accepted* an analysis request since this model's last counts dump - and
it was documented as over-refusing and never under-refusing. That was wrong. Requests are counted when
they are accepted, not when they finish; a counts dump is legal while requests are open; and, more
simply, a position served out of pre-warmed level-0 content never enters the engine's set path at all, so
it earns no position for the evaluations condition either. **A fully pre-warmed card, re-studied with no
new positions - the mature card this feature exists for - could therefore be detached with every one of
its retrievals unwritten, silently.** It is no longer a proxy: the refusal now reads the retrieval counts
themselves. You no longer need to dump counts defensively before detaching a card you only reviewed; the
refusal will tell you.

The response echoes the query and adds:

   * `sourcesDetached (integer)`: files released.
   * `storageReleased (boolean)`: whether the memory actually went. This is **observed, not assumed**: a
     lookup hands out its result in a way that keeps the whole block alive, so a client holding on to a
     result could keep it resident. `false` means that happened.
   * `heapReclaim (string)`: `"trimmed"` if the allocator returned pages to the operating system,
     `"nothingToTrim"` if it had nothing to return, `"unavailable"` on a build whose allocator offers no
     such call (KataGo asks glibc; other C libraries have no equivalent, and this says so rather than
     quietly doing nothing).
   * `discardedUndumpedEntries (integer)`: what `discardUndumped` threw away, or 0.

Detaching frees the pre-warmed level 0. It does not empty the ordinary cache - use `clear_cache` for
that - and it does not unregister the context's name, so you can attach it again later.

#### cache_dump

Writes a context's work to disk. This is the only action that writes anything; nothing is persisted
automatically. Fields:

   * `id (string)`: Required. An arbitrary string identifier for this query.
   * `action (string)`: Required. Should be the string `cache_dump`.
   * `context (string)`: Required. Must be attached.
   * `model (string)`: Optional. Defaults to the model started with `-model`.
   * `what (string)`: **Required**, one of `"counts"`, `"evaluations"` or `"both"`. There is no default:
     which of the two files a write touched is not something to infer from a field you did not send.
   * `admission (object)`: **Required**, exactly one of `{"minLookups":N}` or `{"all":true}`. There is
     no default: a dump writes to disk, and which entries it admits is not something to infer from an
     absent field (SSD wear is a real cost, and picking a currency to gate admission on -- retrievals
     vs. raw presentations vs. deduplicated-per-search presentations -- is deliberately left for the
     operator to decide per dump, not baked in here). `{"minLookups":N}` writes only positions the count
     log records at least `N` retrievals for; `{"minLookups":2}` is the usual "only keep what I have
     seen more than once" policy. `{"all":true}` writes everything this context earned and does not
     already have on disk -- the old default, still reachable, only explicitly.

Example:
```
{"id":"a3","action":"cache_dump","context":"card-5455","what":"both","admission":{"minLookups":2}}
```

With `"both"`, counts are written first and the evaluations are then admitted against the freshly written
counts - so a position first seen and then re-used within this same session can qualify on the strength
of this session, rather than having to wait for the next one.

Dumping the same context twice with no analysis in between writes **nothing** the second time: each
position remembers whether its bytes are already in the file, and retrieval counts are written as
increments since the last dump. Attaching, dumping and detaching repeatedly does not grow your files and
does not inflate your counts.

The response echoes the query and adds `context`, `model`, `openRequestsAtDump`, and:

   * `counts (object)`, when counts were written: `bytesAppended`, `tornTailBytesDiscarded`,
     `rewroteTheFile`, `compacted`, `rowsInLog`, `unattributedLookups`, `tail`.
   * `evaluations (object)`, when evaluations were written: `entriesWritten`, `bytesAppended`,
     `tornTailBytesDiscarded`, `rewroteTheFile`, `compacted`, `markedPersisted`, `admission`, and three
     figures naming everything that was *not* written, separately because they have different remedies:
     `alreadyPersisted` (its bytes are already in the file), `belowThreshold` (your `admission` refused
     it), `notResident` (it was earned but is no longer in the cache, because a capacity sweep dropped
     it).
   * `noAttributableContextEntries`, `unrecordedAttributions (integer)`: how much of the session could not
     be filed under any context. Normally 0; a nonzero figure says the record is short and by how much,
     rather than the work being quietly assigned to whichever context happened to be first.

A dump is legal while analysis requests are open - everything it reads is thread-safe and off the search
path - but the intended posture is to dump at rest, so `openRequestsAtDump` reports how many were open,
and a client that dumped live can see that it did.

**What a counts dump does not write.** Retrievals of a position that no context could be attributed
to - one first computed while several contexts were attached and the request carried no
`cacheContext` - belong to no card's file and are written by none of them. That is the same
population the response's `noAttributableContextEntries` counts, so it is a number you can watch
rather than a silence. Tag your queries with `cacheContext` when more than one context is attached
and it stays at zero.

**Counts are per context, so several cards can be attached at once.** A dump writes exactly the
retrievals of the context you named - served by its own pre-warmed level 0 or by positions it earned -
and advances only those marks, so dumping one attached card leaves every other card's counts whole for
its own dump. An earlier version of this engine refused `counts` outright whenever more than one context
was attached, because the retrieval counts it could reach were kept per cache table and could not be
divided; they are now divided where the facts live, and the refusal is gone.

#### cache_stats

Reports what one model's cache is holding right now. Fields:

   * `id (string)`: Required. An arbitrary string identifier for this query.
   * `action (string)`: Required. Should be the string `cache_stats`.
   * `model (string)`: Optional. Defaults to the model started with `-model`.

Example:
```
{"id":"a4","action":"cache_stats"}
```

The response echoes the query and adds:

   * `residentEntries`, `residentPayloadBytes`, `fixedStructureBytes (integer)`: what is cached and what
     it costs. `capacitySlots (integer)` is the slot count occupancy should be read against, and is 0 for
     a cache configured with `nnCacheCollision = chain`, which is bounded by bytes and has no slot count -
     a 0 here means "not applicable", not "full".
   * `cacheDirectory (string)` and `levelZeroSourcesAttached (integer)`, when `nnCacheDir` is set.
   * `countedKeys`, `retrievalsThisSession`, `unrecordedHits (integer)`: the retrieval counts as they
     stand. These are running totals for reading, not the increments a dump writes; asking for them does
     not consume or change anything.
   * `attributedKeys`, `noAttributableContextEntries`, `unrecordedAttributions (integer)`.
   * `contexts (array of object)`: one per attached context, each with `context`, `levelOneFilled`,
     `levelOneFilledBytes`, `unpersistedEntries` (what a `cache_dump` would owe), `sources` (as
     `cache_attach` reports them), and `countLogRows`, `countLogBlocks`, `countLogTail` read from the
     context's file.

This walks the whole cache under its locks. It is a between-searches reporting call, not something to
poll during analysis.

#### terminate

Requests that KataGo terminate zero or more analysis queries without waiting for them to finish normally. When a query is terminated, the engine will make a best effort to halt their analysis as soon as possible, reporting the results of whatever number of visits were performed up to that point. Required fields:

   * `id (string)`: Required. An arbitrary string identifier for this query.
   * `action (string)`: Required. Should be the string `terminate`.
   * `terminateId (string)`: Required. Terminate queries that were submitted with this `id` field without analyzing or finishing analyzing them.
   * `turnNumbers (array of ints)`: Optional. If provided, restrict only to terminating the queries with that id that were for these turn numbers.

Examples:
```
{"id":"bar","action":"terminate","terminateId":"foo"}
{"id":"bar","action":"terminate","terminateId":"foo","turnNumbers":[1,2]}
```

Responses to terminated queries may be missing their data fields if no analysis at all was performed before termination. In such a case, the only fields guaranteed to be on the response are `id` and `turnNumber`, and `isDuringSearch` (which will always be false), as well as one additional boolean field unique to terminated queries that did not analyze at all, `noResults` (which will always be true). Example:
```
{"id":"foo","isDuringSearch":false,"noResults":true,"turnNumber":2}
```

The terminate query itself will result in a response as well, to acknowledge receipt and processing of the action. The response consists of echoing a json object back with exactly the same fields and data of the query.

The response will NOT generally wait for all of the effects of the action to take place - it may take a small amount of additional time for ongoing searches to actually terminate and report their partial results. A client of this API that wants to wait for all terminated queries to finish should on its own track the set of queries that it has sent for analysis, and wait for all of them to have finished. This can be done by relying on the property that every analysis query, whether terminated or not, and regardless of `reportDuringSearchEvery`, will conclude with exactly one reply where `isDuringSearch` is `false` - such a reply can therefore be used as a marker that an analysis query has finished. (Except during shutdown of the engine if `-quit-without-waiting` was specified).

#### terminate_all

The same as terminate but does not require providing a `terminateId` field and applies to all queries, regardless of their `id`. Required fields:

   * `id (string)`: Required. An arbitrary string identifier for this query.
   * `action (string)`: Required. Should be the string `terminate_all`.
   * `turnNumbers (array of ints)`: Optional. If provided, restrict only to terminating the queries for these turn numbers.

Examples:
```
{"id":"bar","action":"terminate_all"}
{"id":"bar","action":"terminate_all","turnNumbers":[1,2]}
```
The terminate_all query itself will result in a response as well, to acknowledge receipt and processing of the action. The response consists of echoing a json object back with exactly the same fields and data of the query.

See the documentation for terminate above regarding the output from terminated queries. As with terminate, the response to terminate_all will NOT wait for all of the effects of the action to take place, and the results of all the old queries as they are terminated will be reported back asynchronously.

#### query_models

Requests that KataGo report information about the loaded models. Required fields:

   * `id (string)`: Required. An arbitrary string identifier for this query.
   * `action (string)`: Required. Should be the string `query_models`.

Example:
```json
{"id":"foo","action":"query_models"}
```

The response to this query will echo back the same keys passed in, along with a key "models" containing an array of the models loaded. Each model in the array includes details such as the model name, internal name, maximum batch size, whether it uses a human SL profile, version, and FP16 usage. Example:

```json
{
  "id": "foo",
  "action": "query_models",
  "models": [
    {
      "name": "kata1-b18c384nbt-s9732312320-d4245566942.bin.gz",
      "internalName": "kata1-b18c384nbt-s9732312320-d4245566942",
      "maxBatchSize": 256,
      "usesHumanSLProfile": false,
      "version": 14,
      "usingFP16": "auto"
    },
    {
      "name": "b18c384nbt-humanv0.bin.gz",
      "internalName": "b18c384nbt-humanv0",
      "maxBatchSize": 256,
      "usesHumanSLProfile": true,
      "version": 15,
      "usingFP16": "auto"
    }
  ]
}
```

## Human SL Analysis Guide

As of version 1.15.0, released July 2024, KataGo supports a new human supervised learning ("human SL") model `b18c384nbt-humanv0.bin.gz` that was trained on a large number of human games to predict moves by players of all different ranks and the outcomes of those games. People have only just started to experiment with the model and there might be many creative possibilities for analysis or play.

See also the notes on "humanSL" and other parameters within the [GTP human 5k example config](../cpp/configs/gtp_human5k_example.cfg). Although this is a GTP config, not an analysis engine config, the inline documentation about how the "humanSL" parameters behave is just as applicable to the analysis engine.

Similarly, for GTP users, most of the below notes are just as applicable to GTP play and analysis (used by engines like Lizzie or Sabaki) despite being written from the perspective of the analysis engine.

Below are some notes and suggestions for starting points on playing with the human SL model.

### Setting Up to Use the Model

There are two ways to pass in the human SL model.

* The basic intended way: pass an additional argument `-human-model b18c384nbt-humanv0.bin.gz` in addition to still passing in KataGo's normal model.
   * For example: `./katago analysis -config configs/analysis_example.cfg -model models/kata1-b28c512nbt-s7382862592-d4374571218.bin.gz -human-model models/b18c384nbt-humanv0.bin.gz`.
   * Additionally, provide `humanSLProfile` via `overrideSettings` on queries. See documentation above for `overrideSettings`.
   * Additionally, make sure to request `"includePolicy":true` in the query.
   * Then, a new `humanPolicy` field will be reported on the result, indicating KataGo's prediction of how random human players matching the given humanSLProfile (e.g. 5 kyu rank) might play.
   * If no further parameters are set, KataGo's main model will still be used for all other analysis.
   * If further parameters are set, interesting *blended* usages of the KataGo's main model and the human SL model are possible. See some "recipes" below.

* An alternative way: pass `-model b18c384nbt-humanv0.bin.gz` instead of KataGo's normal model, using the human model exclusively.
   * For example: `./katago analysis -config configs/analysis_example.cfg -model models/b18c384nbt-humanv0.bin.gz`.
   * Additionally, provide `humanSLProfile` via `overrideSettings` on queries. See documentation above for `overrideSettings`. (In the case of GTP, set `humanSLProfile` in the GTP config, and update it at runtime via `kata-set-param` if you want to change it dynamically).
   * Then, KataGo will use the human model at the configured profile for all analysis, rather than its normal typically-superhuman analysis.
   * Note that if you are searching with many visits (or even just a few visits!), typically you can expect that KataGo will NOT match the strength of a player of the given humanSLProfile, but will still be stronger because the search will probably solve a lot of tactics that players of a weaker rank would not solve.
      * The human SL model is trained such that using only *one* visit, and full temperature (i.e. choosing random moves from the policy proportionally often, rather than always choosing the top move), will give the closest match to how players of the given rank might play. This should be true up to mid-high dan level, at which point the raw model might start to fall short and need more than 1 visit to keep up in strength.

   * If used as the main model, the human SL model may have significantly more pathologies and biases in its reported winrates and scores than KataGo's normal model, due to the SGF data it trained on.
      * For example, in handicap games it might not report accurate scores or winrates because in recorded human handicap SGFs, White is usually a stronger player than Black and/or some servers may underhandicap games, biasing the result.
      * For example, in even games, it might report erroneous scores and winrates after a big swing or in extreme positions, due to how human players may resign or go on tilt, or due to inaccurately recorded player ranks in the training data, or due to some fraction of sandbagger/airbagger games or AI cheating games in the training data.


### Recipes for Various HumanSL Usages

Here is a brief guide to some example usages, and hopefully a bit of inspiration for possible things to try.

Except for parameters explicitly documented earlier as belonging on the outer json query object (e.g. `includePolicy`, `maxVisits`), the parameters described below should be set within the `overrideSettings` of a query. E.g:

```
"overrideSettings":{"humanSLProfile":"rank_3d","ignorePreRootHistory":false,"humanSLRootExploreProbWeightless":0.5,"humanSLCpuctPermanent":2.0}
```

Do NOT set such parameters as a key of the outer json query object, as that will have no effect. KataGo should issue a warning if you accidentally do. If desired, you can also hardcode parameters within the analysis config file, e.g. `humanSLProfile = rank_3d`.

This guide is also applicable for GTP users, for configuring play and GTP-based analysis (e.g. kata-analyze). For GTP, set parameters within the GTP config file, and optionally change then dynamically via `kata-set-param` ([GTP Extensions](./GTP_Extensions.md)).

#### Human-like play

For simply imitating how a player of a given rank would play, the recommended way is:

* Set `humanSLProfile` appropriately.
* Set `ignorePreRootHistory` to `false` (normally analysis ignores history to be unbiased by move order, but humans definitely behave differently based on recent moves!).
* Send a query with any number of visits (even 1 visit) with `"includePolicy":true` specified on the outer json query object.
* Read `humanPolicy` from the result and pick a random move according to the policy probabilities.

Note that since old historical human games from training might vary in whether they record passes at all, it's possible the human SL net could have trouble passing appropriately in some board positions for some humanSLProfiles. For some weaker ranks, it's possible the human SL net may pass too early and leave positions unfinished in an undesirable way. If so, then the following should work well:

* Set `humanSLProfile` appropriately.
* Set `ignorePreRootHistory` to `false`.
* Send a query with at least a few visits so KataGo can search the position itself (e.g. > 50 visits), still with `"includePolicy":true`.
* If the top moveInfo from the result (the moveInfo with `"order":0`) is a pass, then pass.
* Otherwise, read `humanPolicy` and pick a random move proportional to the policy probabilities, except excluding passing.

(Note: For GTP users, [gtp_human5k_example.cfg](../cpp/configs/gtp_human5k_example.cfg) already does human imitation play by default, with some GTP-specific hacks and parameters to get KataGo's move selection to use the human SL model in the above kind of way. See documentation in that config.)

Optionally, also you can set `rootNumSymmetriesToSample` to `2`, or to `8` instead of the default `1`. This will slightly add latency but improve the quality of the human policy by averaging more symmetries, which might be good when relying so heavily on the raw human policy without any search.

#### Ensuring all likely human moves are analyzed

For analysis and game review, if you want to ensure all moves with high human policy get plenty of visits, you can try settings like the following:

* Set `humanSLProfile` and `ignorePreRootHistory` and `rootNumSymmetriesToSample` as desired.
* Set `humanSLRootExploreProbWeightless` to `0.5` (spend about 50% of playouts to explore human moves, in a weightless way that doesn't bias KataGo's evaluations).
* Set `humanSLCpuctPermanent` to `2.0` or similar (when exploring human moves, ensure high-human-policy moves get many visits even if they lose a lot). Set it to something lower if you want to reduce visits for moves that are judged to be very bad.
* Make sure to use plenty of visits overall.

#### Possible metrics that might be interesting

If you've ensured that all likely human moves are analyzed, there might be some interesting kinds of metrics to consider that can be derived from the human policy:

* Mean score that a player would have after the current move, if sampling from the human policy, `sum(scoreLead * humanPrior) / sum(humanPrior)`.
* Standard deviation of score change due to current move, if sampling from the human policy, `sqrt(sum((scoreLead-m)**2 * humanPrior) / sum(humanPrior))` where m is the above mean.
* Difference in the human policy of the move played between the current rank and a player several ranks higher (send 1-visit queries with other humanSLProfiles to obtain the humanPolicy for other ranks).
* Is something like "(actual score - mean score) / standard deviation of score" an interesting alternative to simply the absolute score loss for judging a mistake?
* Is sorting or weighting mistakes by the amount that a player 4 ranks higher would be less likely to play that move, or other similar kinds of metrics, a good way to bias a game review towards mistakes that are more level-appropriate for a player to review?

#### How to get stronger human-style play

If you want to obtain human *style* moves, but play stronger than a given human level in strength (i.e. match just the style, but not necessarily the strength), or compensate for the gap in strength of the raw neural net at high-dan play, you can try this:

* Ensure all human likely moves are analyzed, as described in an earlier section.
* Choose a random move among all `moveInfos` with probability proportional to `humanPrior * exp(utility / 0.5)`. This will follow the humanPrior, but smoothly attenuate the probability of a move as it starts to lose more than 0.5 utility (about 25% winrate and/or some amount of score). Adjust the divisor 0.5 as desired.
* Optionally, also set `staticScoreUtilityFactor` to `0.5`. (significantly increase how much score affects the utility, compared to just winrate).
* Another significant way to influence the strength is to decrease the temperature settings.
    * In the analysis engine, this would be implemented by additionally raising `humanPrior ** (1/temperature)` for some temperature, and renormalizing.
    * In GTP, this can be done by setting `chosenMoveTemperatureOnlyBelowProb` to `1.0` and then decreasing `chosenMoveTemperatureEarly` and `chosenMoveTemperature`.
    * In either case, decreasing the temperature will make the bot play more deterministically and mimic a narrower fraction of the human distribution, but can improve strength as well.
* A combination of methods like this, with appropriate adjusted numbers, is a good way to compensate for the gap that starts to open up in the human SL model no longer being able to match the strength of very top players at only 1 visit, but experimentation may be needed to tune the numbers.

Note: For GTP users, the parameter `humanSLChosenMovePiklLambda` does precisely this exp-based probability scaling.
An example set of parameters is given in [gtp_human9d_search_example.cfg](../cpp/configs/gtp_human9d_search_example.cfg), some of which may also be instructive to read even if you're using the analysis engine mode rather than GTP.

#### Heavily bias the search to anticipate human-like sequences rather than KataGo sequences.

Not many people have experimented with this yet, but in theory this could have very interesting effects! This will influence the winrates and scores that KataGo assigns to various moves to be much closer to the winrates and scores it would anticipate for various variations if they were played out the way it thinks human players of the configured profile might play them out, but still judging the endpoints of those variations using KataGo's own judgments.

* Set `humanSLProfile` and `ignorePreRootHistory` and `rootNumSymmetriesToSample` as desired.
* Set `humanSLPlaExploreProbWeightful` and `humanSLOppExploreProbWeightful` to `0.9` (spend about 90% of visits at every node using the human policy, in a weightful way that does bias KataGo's evaluations).
* Set `humanSLRootExploreProbWeightful` to `0.5` (at the root use about 50% of playouts to explore human moves, and the other 50% use KataGo's normal policy).
* Set `humanSLCpuctPermanent` to `1.0` or as otherwise desired (when using the human policy, attenuate the policy from too many visits to things that are on the order of 1.0 utility, or 50% winrate worse).
* Set `useUncertainty` to `false` and `subtreeValueBiasFactor` to `0.0` and `useNoisePruning` to `false` (important, disables a few search features that add strength but are highly likely to interfere with this kind of weightful biasing).

#### Bias the search to anticipate human-like sequences rather than KataGo sequences, but only for the opponent.

This kind of setting could be interesting for handicap games or trying to elicit trick plays and other kinds of opponent-aware tactics. Of course, experimentation and tuning may be needed for it to work well, and it might not work well, or might work "too" well and backfire.

* Set `humanSLProfile` and `ignorePreRootHistory` and `rootNumSymmetriesToSample` as desired.
   * This is also probably a really interesting place to experiment with the various `preaz_{BR}_{WR}` or `rank_{BR}_{WR}` settings with asymmetric ranks.
* Set `humanSLOppExploreProbWeightful` to `0.8` (spend about 80% of visits at every node using the human policy, in a weightful way that does bias KataGo's values, but only for the opponent!).
* Set `humanSLCpuctPermanent` to `0.5` or as desired (when using the human policy, do attenuate the policy from putting *too* many visits to things that are on the order of 0.5 utility, or 25% winrate worse).
* Set `playoutDoublingAdvantage` also as desired or as typical for handicap games.
* Set `useUncertainty` to `false` and `subtreeValueBiasFactor` to `0.0` and `useNoisePruning` to `false` (important, disables a few search features that add strength but are highly likely to interfere with this kind of weightful biasing).
   * Setting `useNoisePruning` to `false` is probably the most important of these - it adds the least strength in normal usage but might interfere the most. One could experiment with still enabling the other two for strength.

