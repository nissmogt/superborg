"""Vendor @huggingface/jinja (MIT) as a classic script for the browser engine's Web Worker.

    python wasm/vendor_jinja.py           # downloads the release, checks the pinned hashes, writes wasm/vendor/
    python wasm/vendor_jinja.py --check   # exit 1 when wasm/vendor/hfjinja.js differs from what it would write

wasm/trace-session.js renders chat templates with it (HFJinja.Template), so the browser engine
builds the prompt superborg/chat.py builds. The package's ESM dist has no imports and no
references to host globals; its trailing export block is cut and the body wrapped in a function
that assigns the exports to the global scope (self in a Worker, globalThis in Node), the way
wasm/build-js.sh wraps the wasm2js output. The license text goes in front as a comment, and
wasm/vendor/THIRD_PARTY_NOTICES.md records version, hashes, source and the local patches below.
The generated files are committed: the artifact build does not download anything.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

NAME = "@huggingface/jinja"
VERSION = "0.5.10"
REPO = "https://github.com/huggingface/huggingface.js/tree/main/packages/jinja"
DIST_URL = f"https://cdn.jsdelivr.net/npm/{NAME}@{VERSION}/dist/index.js"
LICENSE_URL = f"https://cdn.jsdelivr.net/npm/{NAME}@{VERSION}/LICENSE"
# pinned from the first download (2026-10-16); a bump of VERSION needs new hashes
DIST_SHA256 = "58cf1a55f9a48154a05bcb8d5b85d0dadfc96aef22fd33f3105ec2151e8bac36"
LICENSE_SHA256 = "d6e7b19451c1b3d1d66353bb4b1274138953b3bfd9ae60c56e77b1f548796982"

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
OUT_JS = VENDOR_DIR / "hfjinja.js"
OUT_NOTICES = VENDOR_DIR / "THIRD_PARTY_NOTICES.md"
EXPORTS = ("Template", "Environment", "Interpreter", "parse", "tokenize")

# Local patches: (name, why, [(old, new), ...]); every `old` must occur exactly once in the dist.
# A version that already carries a fix makes its patch fail to apply, which is the cue to drop it.
_STRIP_HELPER = '''\
// local patch (strip-chars): str.strip/lstrip/rstrip honour their characters argument, as in Python
function stripChars(value, chars, left, right) {
  if (chars === void 0 || chars instanceof NullValue || chars instanceof UndefinedValue) {
    return left && right ? value.trim() : left ? value.trimStart() : value.trimEnd();
  }
  if (!(chars instanceof StringValue)) {
    throw new Error("strip arg must be None or str");
  }
  const set = new Set(Array.from(chars.value));
  const cps = Array.from(value);
  let i = 0, j = cps.length;
  if (left) while (i < j && set.has(cps[i])) i++;
  if (right) while (j > i && set.has(cps[j - 1])) j--;
  return cps.slice(i, j).join("");
}
function strftime_now(format2) {
'''

_STRFTIME_OLD = '''\
function strftime(date, format2) {
  const monthFormatterLong = new Intl.DateTimeFormat(void 0, { month: "long" });
  const monthFormatterShort = new Intl.DateTimeFormat(void 0, { month: "short" });
  const pad2 = (n) => n < 10 ? "0" + n : n.toString();
  return format2.replace(/%[YmdbBHM%]/g, (token) => {
    switch (token) {
      case "%Y":
        return date.getFullYear().toString();
      case "%m":
        return pad2(date.getMonth() + 1);
      case "%d":
        return pad2(date.getDate());
      case "%b":
        return monthFormatterShort.format(date);
      case "%B":
        return monthFormatterLong.format(date);
      case "%H":
        return pad2(date.getHours());
      case "%M":
        return pad2(date.getMinutes());
      case "%%":
        return "%";
      default:
        return token;
    }
  });
}
'''

_STRFTIME_NEW = '''\
// local patch (strftime-c-locale): the directives of Python's datetime.strftime, in the C locale
// (Python never changes the process locale, so its month and day names are English wherever it
// runs), instead of %Y %m %d %b %B %H %M in the browser's locale. Unknown directives stay as they
// are, like glibc; a "-" flag drops the padding (%-d).
var STRFTIME_DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
var STRFTIME_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
  "October", "November", "December"];
function strftime(date, format2) {
  const year = date.getFullYear(), month = date.getMonth(), day = date.getDate(), wday = date.getDay();
  const hours = date.getHours(), minutes = date.getMinutes(), seconds = date.getSeconds();
  // day of the year from two local midnights, so a DST change in between does not shift it
  const yday = Math.round((new Date(year, month, day) - new Date(year, 0, 1)) / 864e5);
  // ISO 8601 week: the year of the week's Thursday, and the week's number in it (Jan 4 is in week 1)
  const thursday = new Date(year, month, day + 3 - ((wday + 6) % 7));
  const isoYear = thursday.getFullYear();
  const week1 = new Date(isoYear, 0, 4);
  week1.setDate(4 + 3 - ((week1.getDay() + 6) % 7));
  const isoWeek = 1 + Math.round((thursday - week1) / 6048e5);
  const num = (n, width, flag, fill = "0") => flag === "-" ? String(n) : String(n).padStart(width, fill);
  const h12 = hours % 12 === 0 ? 12 : hours % 12;
  const out = (token, flag) => {
    switch (token) {
      case "a": return STRFTIME_DAYS[wday].slice(0, 3);
      case "A": return STRFTIME_DAYS[wday];
      case "b": case "h": return STRFTIME_MONTHS[month].slice(0, 3);
      case "B": return STRFTIME_MONTHS[month];
      case "c": return strftime(date, "%a %b %e %H:%M:%S %Y");
      case "C": return num(Math.floor(year / 100), 2, flag);
      case "d": return num(day, 2, flag);
      case "D": return strftime(date, "%m/%d/%y");
      case "e": return num(day, 2, flag, " ");
      case "f": return num(date.getMilliseconds() * 1e3, 6, flag);
      case "F": return strftime(date, "%Y-%m-%d");
      case "g": return num(isoYear % 100, 2, flag);
      case "G": return num(isoYear, 4, flag);
      case "H": return num(hours, 2, flag);
      case "I": return num(h12, 2, flag);
      case "j": return num(yday + 1, 3, flag);
      case "k": return num(hours, 2, flag, " ");
      case "l": return num(h12, 2, flag, " ");
      case "m": return num(month + 1, 2, flag);
      case "M": return num(minutes, 2, flag);
      case "n": return "\\n";
      case "p": return hours < 12 ? "AM" : "PM";
      case "P": return hours < 12 ? "am" : "pm";
      case "R": return strftime(date, "%H:%M");
      case "s": return String(Math.floor(date.getTime() / 1e3));
      case "S": return num(seconds, 2, flag);
      case "t": return "\\t";
      case "T": return strftime(date, "%H:%M:%S");
      case "u": return String(wday === 0 ? 7 : wday);
      case "U": return num(Math.floor((yday + 7 - wday) / 7), 2, flag);
      case "V": return num(isoWeek, 2, flag);
      case "w": return String(wday);
      case "W": return num(Math.floor((yday + 7 - ((wday + 6) % 7)) / 7), 2, flag);
      case "x": return strftime(date, "%m/%d/%y");
      case "X": return strftime(date, "%H:%M:%S");
      case "y": return num(year % 100, 2, flag);
      case "Y": return num(year, 4, flag);
      case "z": case "Z": return "";
      case "%": return "%";
      default: return null;
    }
  };
  return format2.replace(/%(-?)([A-Za-z%])/g, (m, flag, token) => out(token, flag) ?? m);
}
'''

_CONCAT_OLD = '''\
    if (left instanceof StringValue || right instanceof StringValue) {
      switch (node.operator.value) {
        case "+":
          return new StringValue(left.value.toString() + right.value.toString());
      }
    }
'''

_CONCAT_NEW = '''\
    if (left instanceof StringValue || right instanceof StringValue) {
      switch (node.operator.value) {
        case "+":
          // local patch (str-concat): only str + str, as in Python; a dict or list is not turned into text
          if (!(left instanceof StringValue && right instanceof StringValue)) {
            throw new TypeError(`unsupported operand types for +: ${left.type} and ${right.type}`);
          }
          return new StringValue(left.value + right.value);
      }
    }
'''

_NONE_OLD = '''\
      const lastEvaluated = this.evaluate(statement, environment);
      if (lastEvaluated.type !== "NullValue" && lastEvaluated.type !== "UndefinedValue") {
        result += lastEvaluated.toString();
      }
'''

_NONE_NEW = '''\
      const lastEvaluated = this.evaluate(statement, environment);
      if (lastEvaluated.type === "NullValue") {
        // local patch (none-output): {{ expr }} prints None as "None", like str(None) in Python;
        // the statements that evaluate to None (set, macro, comment) print nothing, as before
        if (statement.type !== "Set" && statement.type !== "Macro" && statement.type !== "Comment") {
          result += "None";
        }
      } else if (lastEvaluated.type !== "UndefinedValue") {
        result += lastEvaluated.toString();
      }
'''

PATCHES = [
    ("strip-chars",
     "str.strip(), lstrip() and rstrip() ignore their characters argument (huggingface.js issue #2597, "
     "fix PR #2598 unreleased as of 2026-10-16); Qwen3's template uses rstrip('\\n') and lstrip('\\n') on "
     "reasoning content, so the browser rendered spaces around <think> blocks differently from Python.",
     [("function strftime_now(format2) {\n", _STRIP_HELPER),
      ('        "strip",\n        new FunctionValue(() => {\n          return new StringValue(this.value.trim());\n',
       '        "strip",\n        new FunctionValue((args) => {\n'
       '          return new StringValue(stripChars(this.value, args[0], true, true));\n'),
      ('        "rstrip",\n        new FunctionValue(() => {\n          return new StringValue(this.value.trimEnd());\n',
       '        "rstrip",\n        new FunctionValue((args) => {\n'
       '          return new StringValue(stripChars(this.value, args[0], false, true));\n'),
      ('        "lstrip",\n        new FunctionValue(() => {\n          return new StringValue(this.value.trimStart());\n',
       '        "lstrip",\n        new FunctionValue((args) => {\n'
       '          return new StringValue(stripChars(this.value, args[0], true, false));\n')]),
    ("strftime-c-locale",
     "strftime_now supports only %Y %m %d %b %B %H %M %% and takes month names from the browser's locale "
     "(Intl.DateTimeFormat: '16 Okt 2026' under de-DE), while Python's datetime.strftime uses the C locale "
     "and the full directive set; templates that call strftime_now (SmolLM3 '%d %B %Y', gpt-oss "
     "'%Y-%m-%d', Llama 3.2 '%d %b %Y') must render the same prompt in every browser.",
     [(_STRFTIME_OLD, _STRFTIME_NEW)]),
    ("str-concat",
     "`+` with one str operand turns the other into text (a dict renders as \"[object Map]\"), while Python "
     "raises TypeError for str + dict, list, int or bool; DeepSeek's templates concatenate "
     "tool_calls[].function.arguments into the prompt, so a dict there gave a silently wrong browser prompt "
     "where superborg/server.py answers 400. Only str + str concatenates now.",
     [(_CONCAT_OLD, _CONCAT_NEW)]),
    ("none-output",
     "`{{ expr }}` prints nothing when the expression is None, while Python prints str(None), \"None\" "
     "(an undefined variable prints nothing on both sides); a message whose content is null, as "
     "OpenAI-style clients send tool-call turns, rendered differently in the browser for every template "
     "that prints the content as it is (GLM-4-9B-0414, GLM-5.1). Only expressions print \"None\": the set, "
     "macro and comment statements also evaluate to None and still print nothing.",
     [(_NONE_OLD, _NONE_NEW)]),
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, want: str | None) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310 - fixed https URLs above
        data = r.read()
    got = sha256(data)
    if want is None:
        print(f"{url}: sha256 {got} (pin it)", file=sys.stderr)
    elif got != want:
        raise SystemExit(f"{url}: sha256 {got}, expected {want}")
    return data


def wrap(dist: str, license_text: str) -> str:
    """The classic script: license comment, then the dist body in a function that exports HFJinja."""
    cut = dist.rfind("\nexport {")
    if cut < 0 or not dist.rstrip().endswith("};"):
        raise SystemExit("the dist does not end with an export block")
    body = dist[:cut]
    for name, _why, edits in PATCHES:
        for old, new in edits:
            if body.count(old) != 1:
                raise SystemExit(f"patch {name}: expected text occurs {body.count(old)} times, not once")
            body = body.replace(old, new)
    for bad in ("</script", "<!--", "<script"):
        if bad.lower() in body.lower():
            raise SystemExit(f"the dist contains {bad!r}, which the page inliner refuses")
    if "*/" in license_text:
        raise SystemExit("the license text cannot go in a block comment")
    head = [f"/*! {NAME} {VERSION} - {REPO}",
            f" * {DIST_URL} (sha256 {DIST_SHA256}) as a classic script, with local patches:",
            *[f" *   {name}: see THIRD_PARTY_NOTICES.md" for name, _, _ in PATCHES],
            " * Generated by wasm/vendor_jinja.py; do not edit.", " *"]
    head += [" * " + line if line else " *" for line in license_text.strip().splitlines()]
    head.append(" */")
    return ("\n".join(head) + "\n(function (root) {\n\"use strict\";\n" + body.rstrip("\n") + "\n"
            + "root.HFJinja = { " + ", ".join(EXPORTS) + " };\n"
            + "})(typeof self !== \"undefined\" ? self : globalThis);\n")


def notices(dist: bytes, license_text: str) -> str:
    patches = "\n".join(f"{i}. **{name}**: {why}" for i, (name, why, _) in enumerate(PATCHES, 1))
    return f"""# Third-party notices

## {NAME} {VERSION}

- Package: {NAME} ({REPO}), a JavaScript port of Jinja for chat templates.
- Version: {VERSION}
- Source: {DIST_URL} ({len(dist)} bytes, sha256 `{DIST_SHA256}`); license from {LICENSE_URL}
  (sha256 `{LICENSE_SHA256}`).
- Vendored as `wasm/vendor/hfjinja.js`, generated by `wasm/vendor_jinja.py`: the dist's trailing
  `export` block is removed and the body is wrapped in a function that assigns `HFJinja`
  ({", ".join(EXPORTS)}) to the global scope, so the Web Worker (a classic script started from a
  blob: URL) and Node can load it. `wasm/make_artifact.py` prepends it to the worker bundle.
- Local patches (applied by the generator, in this order):
{patches}
- License: MIT

```
{license_text.strip()}
```
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the committed files instead of writing")
    ap.add_argument("--dist", type=Path, help="an already downloaded dist/index.js (checked against the pinned hash)")
    ap.add_argument("--license", type=Path, help="an already downloaded LICENSE")
    a = ap.parse_args(argv)
    dist = a.dist.read_bytes() if a.dist else fetch(DIST_URL, DIST_SHA256)
    if a.dist and sha256(dist) != DIST_SHA256:
        raise SystemExit(f"{a.dist}: sha256 {sha256(dist)}, expected {DIST_SHA256}")
    lic = a.license.read_bytes() if a.license else fetch(LICENSE_URL, LICENSE_SHA256)
    if a.license and sha256(lic) != LICENSE_SHA256:
        raise SystemExit(f"{a.license}: sha256 {sha256(lic)}, expected {LICENSE_SHA256}")
    js = wrap(dist.decode("utf-8"), lic.decode("utf-8"))
    md = notices(dist, lic.decode("utf-8"))
    if a.check:
        same = OUT_JS.exists() and OUT_JS.read_text() == js and OUT_NOTICES.exists() and OUT_NOTICES.read_text() == md
        print("up to date" if same else "out of date: run wasm/vendor_jinja.py")
        return 0 if same else 1
    VENDOR_DIR.mkdir(exist_ok=True)
    OUT_JS.write_text(js)
    OUT_NOTICES.write_text(md)
    print(f"wrote {OUT_JS} ({len(js)} bytes) and {OUT_NOTICES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
