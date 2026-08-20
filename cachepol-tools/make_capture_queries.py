#!/usr/bin/env python3
"""Build the capture workload for the NN-cache policy sweep: real games, walked turn by turn.

WHY THIS FILE EXISTS -- the failure it replaces
-----------------------------------------------
The previous generator wrote 24 queries of 30 RANDOM SCATTERED MOVES each and searched
each one at 100000 visits, on the reasoning that many distinct positions are what fills a
table.  A 1252-second sweep on a GPU host ran on the resulting trace and could not answer
anything: 1,957,525 lookups produced 136 hits, a hit rate of 6.9e-05, because nothing was
ever asked for twice.

Two mechanisms compounded, and both are properties of the WORKLOAD, not of the engine:

  * No cross-query reuse.  Scattered random positions from different "games" share no
    structure, so a position evaluated for one query is never wanted by another.
  * No intra-search reuse.  A KataGo SearchNode holds its own shared_ptr<NNOutput>, so a
    position evaluated once inside a search tree is never asked of the NN cache again --
    however many visits that search is given.  THE NN CACHE EARNS ITS KEEP ACROSS
    SEARCHES, NEVER WITHIN ONE.

So the knobs invert: MANY queries at MODEST visits over RELATED positions, rather than a
few enormous searches over unrelated ones.  Walking a game turn by turn is exactly that --
consecutive turns share almost their whole search space -- and it is also what a reviewing
GUI does, and what the spaced-repetition use case this cache work exists for looks like.
The shape is not a guess: an earlier capture of 120 queries walking a real 120-move game at
500 visits measured a hit rate of 0.3565 on the same engine and the same cache.

WHERE THE GAMES COME FROM
-------------------------
KataGo plays them, here, at very low visits, through the same analysis protocol the capture
stage already speaks.  No external asset is needed and nothing has to be shipped.  The
sample SGFs under cpp/tests/data are unusable for this -- foxlike.sgf is three moves long --
and any real SGF corpus would be an asset we cannot ship.  `katago match` or `katago
selfplay` would also produce games, and were rejected: both need a second configuration
format and a great deal more apparatus than driving the one protocol this stage already
uses.

Games are made to DIFFER from each other by a seeded random opening of a few moves, after
which the engine plays.  That is deterministic given --seed, so a capture is reproducible,
and it mirrors the reason a review corpus has more than one game in it.

WHAT IT EMITS
-------------
One JSONL query per turn per game, each carrying that game's moves truncated to the turn.
One query per turn rather than one query with many analyzeTurns, because separate queries
are separate searches -- which is the arrangement whose reuse is witnessed above, and the
arrangement a GUI actually produces.
"""

import argparse
import json
import random
import subprocess
import sys

# Go's column letters skip 'I'.  Rows are 1-based from the bottom.
COLS = "ABCDEFGHJKLMNOPQRST"


def vertex(x, y):
    return "%s%d" % (COLS[x], y + 1)


class AnalysisEngine:
    """One long-lived `katago analysis` process, spoken to one query at a time.

    Queries are issued strictly one at a time and each response is read before the next is
    sent, because each move depends on the previous answer.  The engine writes protocol
    responses to stdout and its log to stderr; those are never merged here, for the same
    reason the sweep script never merges them -- a log line in a results stream is how a
    results file becomes unparseable.
    """

    def __init__(self, katago, model, config, logpath):
        self.log = open(logpath, "w")
        self.proc = subprocess.Popen(
            [katago, "analysis", "-config", config, "-model", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )

    def ask(self, query):
        if self.proc.poll() is not None:
            raise RuntimeError(
                "the analysis engine exited (code %s) before answering; its log is beside "
                "the query file" % self.proc.returncode
            )
        self.proc.stdin.write(json.dumps(query) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(
                "the analysis engine closed its output without answering; its log is beside "
                "the query file"
            )
        response = json.loads(line)
        # A protocol-level error is refused loudly here rather than silently producing a
        # shorter game, because a short game is a quiet way to produce a trace with no reuse
        # in it -- exactly the failure this file exists to stop.
        if "error" in response:
            raise RuntimeError("the analysis engine refused a query: %s" % response["error"])
        return response

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.wait(timeout=60)
        self.log.close()


def play_game(engine, gid, args, rng):
    """Play one game: a seeded random opening, then engine moves at --gen-visits."""
    moves = []
    player = "B"
    used = set()
    for _ in range(args.random_opening):
        while True:
            x, y = rng.randrange(args.board_size), rng.randrange(args.board_size)
            if (x, y) not in used:
                used.add((x, y))
                break
        moves.append([player, vertex(x, y)])
        player = "W" if player == "B" else "B"

    while len(moves) < args.turns:
        response = engine.ask(
            {
                "id": "gen-%d-%d" % (gid, len(moves)),
                "moves": moves,
                "rules": args.rules,
                "komi": args.komi,
                "boardXSize": args.board_size,
                "boardYSize": args.board_size,
                "maxVisits": args.gen_visits,
                "analyzeTurns": [len(moves)],
            }
        )
        infos = response.get("moveInfos") or []
        if not infos:
            break
        best = min(infos, key=lambda m: m.get("order", 0))["move"]
        if best in ("pass", "resign"):
            break
        moves.append([player, best])
        player = "W" if player == "B" else "B"
    return moves


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--katago", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True, help="the analysis .cfg used for BOTH phases")
    p.add_argument("--out", required=True, help="where the review queries are written (JSONL)")
    p.add_argument("--log", required=True, help="where the generating engine's own log goes")
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--turns", type=int, default=200, help="moves per game, including the random opening")
    p.add_argument("--gen-visits", type=int, default=8, help="visits per move while GENERATING a game")
    p.add_argument("--visits", type=int, default=500, help="visits per REVIEW query -- modest on purpose")
    p.add_argument("--random-opening", type=int, default=6)
    p.add_argument("--ownership", type=int, default=1, choices=[0, 1])
    p.add_argument("--board-size", type=int, default=19)
    p.add_argument("--komi", type=float, default=7.5)
    p.add_argument("--rules", default="tromp-taylor")
    p.add_argument("--seed", type=int, default=20260818)
    args = p.parse_args()

    rng = random.Random(args.seed)
    engine = AnalysisEngine(args.katago, args.model, args.config, args.log)
    games = []
    try:
        for gid in range(args.games):
            moves = play_game(engine, gid, args, rng)
            games.append(moves)
            sys.stderr.write("make_capture_queries: game %d generated, %d moves\n" % (gid, len(moves)))
    finally:
        engine.close()

    queries = 0
    with open(args.out, "w") as f:
        for gid, moves in enumerate(games):
            # Turn 0 (the empty board) through turn len(moves): the whole walk a reviewer
            # makes, in order, one search each.
            for turn in range(len(moves) + 1):
                q = {
                    "id": "rev-%d-%d" % (gid, turn),
                    "moves": moves[:turn],
                    "rules": args.rules,
                    "komi": args.komi,
                    "boardXSize": args.board_size,
                    "boardYSize": args.board_size,
                    "maxVisits": args.visits,
                    "analyzeTurns": [turn],
                }
                if args.ownership:
                    q["includeOwnership"] = True
                f.write(json.dumps(q) + "\n")
                queries += 1

    total_moves = sum(len(m) for m in games)
    if queries == 0:
        raise SystemExit(
            "make_capture_queries: refusing to write an empty query file. No game was generated, "
            "so there is nothing to review and any trace taken from this would have no reuse in it."
        )
    sys.stderr.write(
        "make_capture_queries: %d games, %d moves total, %d review queries at %d visits each "
        "(~%d visits in the whole capture)\n"
        % (len(games), total_moves, queries, args.visits, queries * args.visits)
    )


if __name__ == "__main__":
    main()
