"""Generate engine/unicode_tables.h: the character classes the tokenizer's pre-tokenizers need.

The classes must match the ones Hugging Face `tokenizers` uses, which are not Python's:
  * the pre-tokenizer regexes run on Oniguruma, whose \\p{L}, \\p{N} and \\s follow
    Unicode 16.0 in current releases (tokenizers 0.2x);
  * the Digits pre-tokenizer uses Rust's char::is_numeric, which follows the Unicode
    version of the Rust standard library the wheel was built with (17.0).
Python's unicodedata lags behind (Python 3.13 ships 15.1), so the tables are built from the
UnicodeData.txt of the pinned versions (downloaded once into --cache). Pass --offline to use
the running Python's unicodedata instead (fewer letters/digits in new scripts).

The versions were found by probing `tokenizers` itself over every codepoint; --verify repeats
that probe against the installed `tokenizers` (0.23.3 agrees on all 1,112,064 codepoints).

Usage: python3 tools/gen_unicode_tables.py [--out engine/unicode_tables.h] [--verify]
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

REGEX_UNICODE = "16.0.0"    # Oniguruma inside tokenizers
NUMERIC_UNICODE = "17.0.0"  # Rust char::is_numeric inside tokenizers

URL = "https://www.unicode.org/Public/{v}/ucd/UnicodeData.txt"


def load_categories(version: str, cache: str, offline: bool) -> tuple[str, dict[int, str]]:
    """Return (version, {codepoint: general category}) for every assigned codepoint."""
    if offline:
        import unicodedata
        cats = {}
        for c in range(0x110000):
            gc = unicodedata.category(chr(c))
            if gc != "Cn":
                cats[c] = gc
        return unicodedata.unidata_version, cats
    path = os.path.join(cache, f"UnicodeData-{version}.txt")
    if not os.path.exists(path):
        os.makedirs(cache, exist_ok=True)
        with urllib.request.urlopen(URL.format(v=version), timeout=60) as r:
            data = r.read()
        with open(path + ".tmp", "wb") as f:
            f.write(data)
        os.replace(path + ".tmp", path)
    cats: dict[int, str] = {}
    first = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = line.split(";")
            if len(fields) < 3:
                continue
            cp, name, gc = int(fields[0], 16), fields[1], fields[2]
            if name.endswith(", First>"):
                first = cp
                continue
            if name.endswith(", Last>"):  # ranges such as CJK ideographs are given by their ends
                for c in range(first, cp + 1):
                    cats[c] = gc
                first = None
                continue
            cats[cp] = gc
    return version, cats


def ranges(cps) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for c in sorted(cps):
        if out and out[-1][1] == c - 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return [(a, b) for a, b in out]


def probe_tokenizers() -> dict[str, set[int]]:
    """The classes as the installed Hugging Face `tokenizers` sees them."""
    from tokenizers import Regex, pre_tokenizers
    cps = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
    text = "".join(map(chr, cps))
    out = {}
    for name, pattern in (("L", r"\p{L}"), ("N", r"\p{N}"), ("S", r"\s")):
        # "removed" keeps only what the pattern does not match
        kept = pre_tokenizers.Split(Regex(pattern), behavior="removed").pre_tokenize_str(text)
        rest = {ord(ch) for piece, _ in kept for ch in piece}
        out[name] = set(cps) - rest
    # Digits isolates numeric characters: they come back as single-character pieces
    sep = "a"
    pieces = pre_tokenizers.Digits(individual_digits=True).pre_tokenize_str(sep + sep.join(map(chr, cps)) + sep)
    out["numeric"] = {ord(p) for p, _ in pieces if len(p) == 1 and p != sep}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "..", "engine", "unicode_tables.h"))
    ap.add_argument("--cache", default=os.path.join(os.path.expanduser("~"), ".cache", "superborg", "ucd"))
    ap.add_argument("--offline", action="store_true", help="use Python's unicodedata instead of downloading")
    ap.add_argument("--verify", action="store_true", help="compare with the installed `tokenizers` package")
    a = ap.parse_args(argv)

    rver, rcats = load_categories(REGEX_UNICODE, a.cache, a.offline)
    nver, ncats = load_categories(NUMERIC_UNICODE, a.cache, a.offline)

    # Oniguruma: \p{L} = L*, \p{N} = N*, \s = 0009-000D, 0085, Zs, Zl, Zp (= White_Space).
    letters = [c for c, gc in rcats.items() if gc[0] == "L"]
    numbers = [c for c, gc in rcats.items() if gc[0] == "N"]
    spaces = [c for c, gc in rcats.items() if gc in ("Zs", "Zl", "Zp")] + [9, 10, 11, 12, 13, 0x85]
    numeric = [c for c, gc in ncats.items() if gc in ("Nd", "Nl", "No")]

    classes = [(lo, hi, "UC_L") for lo, hi in ranges(letters)]
    classes += [(lo, hi, "UC_N") for lo, hi in ranges(numbers)]
    classes += [(lo, hi, "UC_S") for lo, hi in ranges(set(spaces))]
    classes.sort()
    for (_, hi, _), (lo, _, _) in zip(classes, classes[1:]):
        assert hi < lo, "overlapping classes"
    numeric_r = ranges(numeric)

    if a.verify:
        got = probe_tokenizers()
        ok = True
        for name, mine in (("L", letters), ("N", numbers), ("S", spaces), ("numeric", numeric)):
            diff = set(mine) ^ got[name]
            if diff:
                ok = False
                print(f"{name}: {len(diff)} codepoints differ from tokenizers, e.g. "
                      f"{[hex(c) for c in sorted(diff)[:10]]}", file=sys.stderr)
        if not ok:
            return 1
        print("tables agree with the installed tokenizers package", file=sys.stderr)

    lines = [
        "/* Generated by tools/gen_unicode_tables.py -- do not edit. */",
        f"/* Character classes of the pre-tokenizer regexes (Oniguruma, Unicode {rver}):",
        " * UC_L = \\p{L}, UC_N = \\p{N}, UC_S = \\s (White_Space); anything else is UC_O.",
        f" * borg_uc_numeric: Rust char::is_numeric (Unicode {nver}), used by the Digits pre-tokenizer. */",
        "#ifndef BORG_UNICODE_TABLES_H",
        "#define BORG_UNICODE_TABLES_H",
        "",
        "#include <stdint.h>",
        "",
        "enum { UC_O = 0, UC_L = 1, UC_N = 2, UC_S = 3 };",
        "",
        "typedef struct {",
        "    uint32_t lo, hi; /* inclusive */",
        "    uint8_t cls;",
        "} borg_uc_range;",
        "",
        f"#define BORG_UC_NCLASS {len(classes)}",
        "static const borg_uc_range borg_uc_class[BORG_UC_NCLASS] = {",
    ]
    row = []
    for lo, hi, cls in classes:
        row.append(f"{{0x{lo:X}, 0x{hi:X}, {cls}}},")
        if len(row) == 4:
            lines.append("    " + " ".join(row))
            row = []
    if row:
        lines.append("    " + " ".join(row))
    lines += ["};", "", f"#define BORG_UC_NNUMERIC {len(numeric_r)}",
              "static const uint32_t borg_uc_numeric[BORG_UC_NNUMERIC][2] = {"]
    row = []
    for lo, hi in numeric_r:
        row.append(f"{{0x{lo:X}, 0x{hi:X}}},")
        if len(row) == 6:
            lines.append("    " + " ".join(row))
            row = []
    if row:
        lines.append("    " + " ".join(row))
    lines += ["};", "", "#endif /* BORG_UNICODE_TABLES_H */", ""]
    with open(a.out, "w", encoding="ascii") as f:
        f.write("\n".join(lines))
    print(f"wrote {a.out}: {len(classes)} class ranges (Unicode {rver}), "
          f"{len(numeric_r)} numeric ranges (Unicode {nver})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
