"""Driving one analysis-engine process from outside it, and reading what it did.

WHY THIS IS A MODULE AND NOT COPIED INTO EACH WITNESS. Two things here are FACTS ABOUT THE
ENGINE rather than conveniences, and a second hand-written copy of either is a second author
of one truth (ADR-0012 P1/P7):

  * The request/response discipline. The analysis protocol is ASYNCHRONOUS -- a query is
    queued and answered later -- so a driver that writes every line at once and reads the
    output afterwards is not sending "attach, analyze, dump", it is sending all three while
    the analysis is still running. A witness whose legs are about what happens at rest needs
    a client that waits for each response before sending the next line, and that discipline
    has to be the same one in every witness or two witnesses' legs mean different things.

  * The end-of-run row counter's LOG FORMAT. The engine writes, per hosted model, the model
    file name and then "NN rows: N" on the following line (cpp/command/analysis.cpp, the
    per-hosted-model loop at the end of the run). Reading it is reading the engine's own
    accounting of how much neural net work it actually did. The pairing of the two lines is
    the format, and the format lives here once.

WHAT IT DELIBERATELY DOES NOT DO. It asserts nothing and knows no leg. A witness decides what
is true; this decides only how to ask.
"""

import json
import re
import subprocess


def engine_args(binary, config, models):
  """The command line that hosts `models[0]` plus every later entry as an extra model."""
  args = [binary, "analysis", "-config", config, "-model", models[0]]
  for extra in models[1:]:
    args += ["-extra-model", extra]
  return args


class EngineClosedStdout(RuntimeError):
  """The engine's stdout ended while a response was still owed."""

  def __init__(self, wanted):
    super().__init__("engine closed stdout while waiting for id %r" % (wanted,))
    self.wanted = wanted


class Engine:
  """One analysis-engine process, driven the way a real client drives it.

  `ask` sends one request and returns the FIRST response carrying its id, so that a leg about
  what happens with nothing in flight really has nothing in flight. `send` is the deliberate
  opposite, for a leg about what happens while something IS.
  """

  def __init__(self, binary, config, models, timeout=1800):
    self.proc = subprocess.Popen(
      engine_args(binary, config, models),
      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    self.timeout = timeout
    self.lines = []

  def send(self, obj):
    self.proc.stdin.write(json.dumps(obj) + "\n")
    self.proc.stdin.flush()

  def ask(self, obj):
    self.send(obj)
    want = obj["id"]
    while True:
      line = self.proc.stdout.readline()
      if line == "":
        raise EngineClosedStdout(want)
      line = line.strip()
      if not line:
        continue
      self.lines.append(line)
      got = json.loads(line)
      if got.get("id") == want:
        return got

  def finish(self):
    """Closes stdin, waits for exit, and returns (returncode, stdout-so-far, stderr)."""
    self.proc.stdin.close()
    rest, err = self.proc.communicate(timeout=self.timeout)
    for line in rest.splitlines():
      if line.strip():
        self.lines.append(line.strip())
    return self.proc.returncode, "\n".join(self.lines) + "\n", err

  def kill(self):
    """For a process a leg abandoned. Safe to call after finish()."""
    if self.proc.poll() is None:
      self.proc.kill()
      try:
        self.proc.communicate(timeout=30)
      except subprocess.TimeoutExpired:
        pass


def run_engine(binary, config, models, lines, timeout=1800):
  """Every line at once, for a leg that is ABOUT not waiting. Returns (rc, stdout, stderr)."""
  proc = subprocess.run(
    engine_args(binary, config, models),
    input="".join(json.dumps(line) + "\n" for line in lines),
    capture_output=True, text=True, timeout=timeout,
  )
  return proc.returncode, proc.stdout, proc.stderr


def responses(stdout):
  """Every response line, keyed by its "id" (a general error has none, keyed under "")."""
  out = {}
  for line in stdout.splitlines():
    line = line.strip()
    if not line:
      continue
    obj = json.loads(line)
    out.setdefault(obj.get("id", ""), []).append(obj)
  return out


# The engine writes, at the end of a run and once per hosted model, the model's FILE NAME and
# then its counters on the following lines. The file name is what the command line named, so a
# caller that started "-model /x/14.gz -extra-model /x/18.gz" reads those two paths back.
_MODEL_LINE = re.compile(r"^(?:.*?: )?(\S+\.(?:gz|bin|txt))$")
_ROWS_LINE = re.compile(r"^(?:.*?: )?NN rows: (\d+)$")


def nn_rows_by_model(stderr):
  """{model file name: rows the neural net actually evaluated}, from the engine's own log.

  THE OBSERVATION THIS EXISTS FOR. A response field saying "N entries attached" is the
  engine's opinion of itself; this is the work that did or did not happen. It is attributed
  PER MODEL because a witness that hosts two nets and sums them cannot tell "the second net
  did no work" from "the first net did it instead" (ADR-0021 Rule 1: observe the property, at
  the site of the claim).

  Raises RuntimeError if a "NN rows:" line is met with no model line before it, rather than
  silently attributing it to the previous model or dropping it.
  """
  rows = {}
  current = None
  for raw in stderr.splitlines():
    line = raw.strip()
    m = _MODEL_LINE.match(line)
    if m:
      current = m.group(1)
      continue
    m = _ROWS_LINE.match(line)
    if m:
      if current is None:
        raise RuntimeError("a 'NN rows:' line with no model file name before it: %r" % (raw,))
      rows[current] = rows.get(current, 0) + int(m.group(1))
      current = None
  return rows


def nn_rows_for(stderr, model_path):
  """The rows one model evaluated. Raises if the engine never reported that model at all --
  an absent counter is not zero work, it is a run whose accounting was not read."""
  rows = nn_rows_by_model(stderr)
  if model_path not in rows:
    raise RuntimeError(
      "the engine reported no 'NN rows:' for %r; it reported %r" % (model_path, sorted(rows))
    )
  return rows[model_path]
