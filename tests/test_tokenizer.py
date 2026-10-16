"""Token-for-token parity of the borg tokenizer (engine/tokenizer.c) with Hugging Face `tokenizers`.

Environment:
  BORG_LIB         the engine shared library (default build/libborg.so)
  BORG_MODELS_DIR  directory of .borg files; one file per tokenizer (the smallest) is tested
  HF_MODELS_DIR    directory of Hugging Face model folders holding tokenizer.json; a .borg file's
                   folder is found from its header "source" ("org/name" -> "org_name" or "name"),
                   falling back to the local Hugging Face hub cache
  BORG_TOK_RANDOM  number of random strings (default 3000)

Run as a script (python tests/test_tokenizer.py) to print per-model parity counts and throughput.
"""
from __future__ import annotations

import ctypes
import json
import os
import random
import struct
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH = os.environ.get("BORG_LIB", os.path.join(ROOT, "build", "libborg.so"))
MODELS_DIR = os.environ.get("BORG_MODELS_DIR", "")
HF_DIR = os.environ.get("HF_MODELS_DIR", "")
N_RANDOM = int(os.environ.get("BORG_TOK_RANDOM", "3000"))

tokenizers = pytest.importorskip("tokenizers")


# ---------------------------------------------------------------------------
# engine binding

class Borg:
    def __init__(self, path: str):
        lib = ctypes.CDLL(path)
        lib.borg_model_load.restype = ctypes.c_void_p
        lib.borg_model_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        lib.borg_model_free.argtypes = [ctypes.c_void_p]
        lib.borg_tokenize.restype = ctypes.c_int
        lib.borg_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        lib.borg_detokenize.restype = ctypes.c_int
        lib.borg_detokenize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
                                        ctypes.c_char_p, ctypes.c_int]
        lib.borg_token_bytes.restype = ctypes.c_int
        lib.borg_token_bytes.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int]
        lib.borg_token_type.restype = ctypes.c_int
        lib.borg_token_type.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.borg_is_eog.restype = ctypes.c_int
        lib.borg_is_eog.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        self.lib = lib

    def load(self, path: str) -> int:
        err = ctypes.create_string_buffer(512)
        m = self.lib.borg_model_load(path.encode(), err, len(err))
        if not m:
            raise RuntimeError(f"borg_model_load({path}): {err.value.decode(errors='replace')}")
        return m

    def tokenize(self, m: int, data: bytes, add_bos: int = 0, special: int = 1) -> list[int]:
        cap = len(data) + 8
        buf = (ctypes.c_int32 * cap)()
        n = self.lib.borg_tokenize(m, data, len(data), add_bos, special, buf, cap)
        if n < 0:
            raise AssertionError(f"borg_tokenize needs {-n} > {cap} slots")
        return list(buf[:n])

    def detokenize(self, m: int, ids: list[int], skip_special: int = 0) -> bytes:
        arr = (ctypes.c_int32 * max(1, len(ids)))(*ids)
        cap = 16 + 4 * len(ids)
        while True:
            buf = ctypes.create_string_buffer(cap)
            n = self.lib.borg_detokenize(m, arr, len(ids), skip_special, buf, cap)
            if n <= cap:
                return buf.raw[:n]
            cap = n

    def token_bytes(self, m: int, tok: int) -> bytes:
        buf = ctypes.create_string_buffer(256)
        n = self.lib.borg_token_bytes(m, tok, buf, 256)
        if n > 256:
            buf = ctypes.create_string_buffer(n)
            self.lib.borg_token_bytes(m, tok, buf, n)
        return buf.raw[:n]


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        magic, _version, hlen = struct.unpack("<4sIQ", f.read(16))
        if magic != b"BORG":
            raise ValueError(f"{path} is not a .borg file")
        return json.loads(f.read(hlen))


def find_tokenizer_json(source: str) -> str | None:
    names = [source.replace("/", "_"), source.split("/")[-1], source]
    for d in filter(None, [HF_DIR]):
        for n in names:
            p = os.path.join(d, n, "tokenizer.json")
            if os.path.exists(p):
                return p
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(source, "tokenizer.json")
        if isinstance(p, str) and os.path.exists(p):
            return p
    except Exception:
        pass
    return None


def discover() -> list[tuple[str, str, str]]:
    """(source, .borg path, tokenizer.json path): the smallest .borg file of every tokenizer."""
    if not MODELS_DIR or not os.path.isdir(MODELS_DIR):
        return []
    best: dict[str, tuple[int, str, str]] = {}
    for fn in sorted(os.listdir(MODELS_DIR)):
        if not fn.endswith(".borg"):
            continue
        path = os.path.join(MODELS_DIR, fn)
        try:
            src = read_header(path).get("source", fn)
        except (OSError, ValueError, struct.error):
            continue
        tj = find_tokenizer_json(src)
        if not tj:
            continue
        key = os.path.realpath(tj)
        size = os.path.getsize(path)
        if key not in best or size < best[key][0]:
            best[key] = (size, path, src)
    return sorted(((src, path, tj) for tj, (_, path, src) in best.items()), key=lambda x: x[0])


# ---------------------------------------------------------------------------
# corpus

PROSE = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello, world! How are you doing today?",
    "In 1969, Neil Armstrong became the first person to walk on the Moon. He said: \"That's one small step for "
    "[a] man, one giant leap for mankind.\"",
    "It's a truth universally acknowledged, that a single man in possession of a good fortune, must be in want "
    "of a wife.",
    "I'LL be there; YOU'RE late, WE'VE waited, she'd said, they'll go, I'm fine, isn't it? DON'T. Y'ALL'D've.",
    "it's IT'S It'S iT's 'tis 'TWAS rock 'n' roll o'clock ma'am '' ''' 's 't 're 've 'm 'll 'd 'x 'S 'T 'RE",
    "Mr. O'Neil's cat's toys' boxes—weren't they ‘quoted’ or “double-quoted”? «Guillemets» „Gänsefüßchen“.",
    "Prices: $5, $10.99, €1.000,50, £3, ¥500, 100%, 3.14159, 2.718e-10, 1/2, ½, ¾, 10^6, 0x1F, 0b1010, 1_000_000.",
    "Call me at +1 (555) 123-4567 or 555.123.4567 on 2024-01-15T08:30:00Z.",
    "Numbers: 1 12 123 1234 12345 123456 1234567 12345678 123456789 1234567890 12345678901234567890",
    "1,234,567 and 1 234 567 and 1.234.567 and 007 and 0 and 00 and 000 and 0000.",
    "ÀÉÎÕÜ àéîõü çÇ ñÑ ßẞ ÆæØøÅå Œœ ĳĲ ﬁﬂ ſ ǅ ǈ ǋ",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et "
    "dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
    "   leading spaces", "trailing spaces   ", "  both  ", " ", "  ", "   ", "\t", "\n", "\r\n", "\n\n\n",
    " \n ", "\t\t\tindented", "a\tb\tc", "line1\nline2\r\nline3\rline4", "end with newline\n",
    "multiple   spaces   between   words", "tab\t \tmix \t", "spaces before newline   \nnext",
    "\n   \n\t\n", "x \n\n y", "word nbsp  twice", "ideographic　space　　",
    "en em thin hair zw​zwnj‌zwj‍", "line para next\u0085nel",
    "vt\x0bff\x0cfs\x1cgs\x1drs\x1eus\x1f", "nul\x00byte", "bell\x07esc\x1b[0m del\x7f",
    "", "a", "A", "1", ".", "'", "''", "' '", "  'quoted'  ",
    "!!!", "...", "?!?!", "---", "___", "***bold***", "~~~", "<<>>", "((()))", "[[]]", "{{}}", "@#$%^&*",
    "....!!!!????", "-->", "<!-- comment -->", "=>", "!==", "::", "->", "...\n\n", "!\r\n\r\nx",
    "▁already▁has▁meta▁space", "▁", "▁▁ ▁",
]

CODE = [
    "def fibonacci(n: int) -> int:\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
    "import numpy as np\n\nx = np.arange(10).reshape(2, 5)\nprint(x.T @ x)  # matrix product\n",
    "class Foo(Bar):\n\tdef __init__(self, *args, **kwargs):\n\t\tsuper().__init__(*args, **kwargs)\n\t\tself._x = {}\n",
    "#include <stdio.h>\n\nint main(int argc, char **argv) {\n    for (int i = 0; i < argc; i++)\n"
    "        printf(\"%d: %s\\n\", i, argv[i]);\n    return 0;\n}\n",
    "static inline uint32_t hash(const uint8_t *p, size_t n) { uint32_t h = 2166136261u; while (n--) h = (h ^ *p++)"
    " * 16777619u; return h; }",
    '{"name": "superborg", "version": "0.1.0", "tags": ["llm", "inference"], "nested": {"a": [1, 2.5, -3e8, null,'
    ' true, false]}, "unicode": "\\u00e9\\ud83d\\ude00"}',
    "<html>\n  <body class=\"main\">\n    <p>Hello &amp; welcome &lt;3</p>\n  </body>\n</html>",
    "SELECT u.id, COUNT(*) AS n FROM users u JOIN orders o ON o.user_id = u.id WHERE u.created_at >= '2024-01-01'"
    " GROUP BY u.id HAVING n > 10 ORDER BY n DESC;",
    "fn main() {\n    let v: Vec<i32> = (1..=10).filter(|x| x % 2 == 0).collect();\n    println!(\"{:?}\", v);\n}",
    "    \n        \n    x = 1\n\n\n\n    y = 2    \n",
    "if (a&&b||!c) { x+=1; y-=2; z*=3; w/=4; } else if(x>=y&&y<=z){return x<<2|y>>1;}",
    "$ ls -la /usr/local/bin | grep -E '^d' && echo \"done\" > /dev/null 2>&1",
    "```python\nprint('hi')\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n",
    "https://example.com/path/to/page?query=1&other=two#fragment and mailto:someone@example.org",
    "// TODO(user): fix this\n/* multi\n * line\n */\n# comment\n-- sql comment\n% latex\n; lisp",
    "\\begin{equation} E = mc^2 \\quad \\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2} \\end{equation}",
]

MULTILINGUAL = [
    "日本語のテキストです。東京は日本の首都です。カタカナとひらがな、そして漢字。",
    "中文文本测试：你好，世界！我们今天去北京。繁體中文：臺灣、香港。",
    "한국어 텍스트입니다. 안녕하세요, 세계! 서울은 대한민국의 수도입니다.",
    "العربية: مرحبا بالعالم! هذه جملة باللغة العربية مع أرقام ١٢٣٤٥٦٧٨٩٠ وتشكيل: مَرْحَبًا",
    "עברית: שלום עולם! זה משפט בעברית.",
    "हिन्दी: नमस्ते दुनिया! यह एक वाक्य है। संख्याएँ: १२३४५६७८९०",
    "বাংলা: হ্যালো বিশ্ব। தமிழ்: வணக்கம் உலகம். తెలుగు: హలో ప్రపంచం. ಕನ್ನಡ ಮಲಯാളം ગુજરાતી ਪੰਜਾਬੀ",
    "ภาษาไทย: สวัสดีชาวโลก นี่คือประโยคภาษาไทย ๑๒๓๔๕",
    "ລາວ ພາສາລາວ, ខ្មែរ ភាសាខ្មែរ, မြန်မာ မြန်မာဘာသာ, བོད་ཡིག",
    "Русский: Привет, мир! Это предложение на русском языке. ЁЛКИ-ПАЛКИ, ёжик.",
    "Українська: Привіт, світе! Ґанок, їжак, є. Български: Здравей, свят! Српски: Здраво свете!",
    "Ελληνικά: Γειά σου Κόσμε! Αυτή είναι μια πρόταση. ΑΒΓΔ αβγδ ς σ Σ",
    "Tiếng Việt: Xin chào thế giới! Đây là một câu tiếng Việt có dấu: ắ ằ ẳ ẵ ặ.",
    "Türkçe: Merhaba dünya! İstanbul'da ılık bir gün. Polski: Zażółć gęślą jaźń. Čeština: Příliš žluťoučký kůň.",
    "Deutsch: Größenwahn, Straße, Übermäßig. Français: Ça va? L'été, c'est où? Español: ¿Qué tal? ¡Olé!",
    "Íslenska: Þetta er ð. Magyar: Árvíztűrő tükörfúrógép. Română: Șși Țț. Lietuvių: ąčęėįšųūž.",
    "ქართული: გამარჯობა. Հայերեն: Բարեւ. አማርኛ: ሰላም ልዑል. ᏣᎳᎩ. ᐃᓄᒃᑎᑐᑦ. ᚠᚢᚦᚨᚱᚲ. 𐌰𐌱𐌲 Gothic.",
    "Emoji: 😀😃😄😁 👍🏻👍🏼👍🏽👍🏾👍🏿 👨‍👩‍👧‍👦 👩🏽‍💻 🏳️‍🌈 🇺🇸🇯🇵🇪🇺 ❤️ ✨ 🧑‍🤝‍🧑 1️⃣ #️⃣ 🫠🪿",
    "Combining: é à ỗ ñ Z̵̶̷a̸̹l̺̻g̼̽o̾",
    "Math: ∀x∈ℝ, ∃y: x²+y²=z² ⇒ ∑ᵢ₌₁ⁿ i = n(n+1)/2 ≈ ∞ ≠ ∅; √2 ∉ ℚ; 𝔸𝔹ℂ 𝒜ℬ𝒞 𝟘𝟙𝟚",
    "Fullwidth: ＡＢＣ ａｂｃ １２３ ！？ Halfwidth: ｱｲｳｴｵ",
    "Roman numerals Ⅰ Ⅱ Ⅲ Ⅳ ⅰⅱⅲ, circled ① ② ③ ⑩ ⑳, superscripts ¹²³⁴, fractions ⅓ ⅔ ⅛",
    "Mixed: Hello世界123abc!@#مرحباПривет🙂こんにちは",
    "Zero width: a​b‌c‍d﻿e⁠f soft­hyphen",
    "Bidi: ‪LRE‬ ‫RLE‬ ⁦LRI⁩ ‎LRM ‏RLM",
    "Private use:  \U000f0000 \U0010fffd, noncharacters ￾￿ \U0001fffe",
    "Rare scripts: 𓀀𓀁𓀂 Egyptian, 𒀀𒀁 Cuneiform, 𐤀𐤁 Phoenician, ꦲꦏ꧀ꦱꦫ Javanese, ᠮᠣᠩᠭᠣᠯ Mongolian",
    "Unicode 15/16/17: \U00011f50\U00011f51 Kawi digits, \U0001e4f0 Nag Mundari, Ᲊᲊ, \U000105c0\U000105c1 "
    "Todhri, \U00016d40\U00016d41 Kirat Rai, \U00011de0\U00011de1 Tolong Siki digits, \U00016ff4",
    "Long s contractions: 'ſ 'ſt it'ſ IT'Ś x'K y'K",
    "Digits in scripts: ٠١٢٣ ۰۱۲۳ ߀߁߂ ०१२ ০১২ ੦੧੨ ૦૧૨ ୦୧୨ ௦௧௨ ౦౧౨ ೦೧೨ ൦൧൨ ๐๑๒ ໐໑໒ ༠༡༢ ၀၁၂ ០១២ ᠐᠑᠒ ０１２",
]

CHAT = [
    "<|im_start|>user\nhi<|im_end|>",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
    "<|im_start|>assistant\n",
    "<|im_start|>assistant\n<think>\nLet me think.\n</think>\n\nThe answer is 4.<|im_end|>",
    "<|endoftext|>", "<|endoftext|><|endoftext|>", "text<|endoftext|>more text", " <|im_end|> ", "<|im_end",
    "<|im_start|><|im_end|><|im_start|>", "<|im_start|>  spaced  <|im_end|>", "x<|im_start|>y",
    "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>", "<tool_response>ok</tool_response>",
    "<|fim_prefix|>def f():<|fim_suffix|>\n<|fim_middle|>", "<|vision_start|><|image_pad|><|vision_end|>",
    "<s>[INST] hello [/INST]", "<s>hi", "<s> hi", "hi</s>", "<s></s><unk>", "<s><s>", "a<s>b</s>c", " <s> ",
    "<|user|>\nhello</s>\n<|assistant|>\n", "<|system|>\nBe brief.</s>\n<|user|>\nWho?</s>\n<|assistant|>\n",
    "<repo_name>borg<file_sep>main.c<filename>x<gh_stars>5", "<jupyter_code>print(1)<jupyter_output>1<empty_output>",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>", "<<|im_start|>>", "<|<|im_end|>|>",
    "<think></think>", "</think>", "<think>", "<|im_start|>user\nI'LL write 123 lines　now<|im_end|>\n",
    "it'<|im_end|>s", "'<|im_end|>ll", "x <|im_end|>  y", "  <|im_start|>", "\n<|im_start|>\n", "1<|im_end|>2",
    "hello  <|im_end|>", "<|im_end|>  hello", "a　<|im_end|>　b", "<s>  </s>", "<s>\n", " <s>", "</s> ",
]

FRAGMENTS = (PROSE + CODE + MULTILINGUAL + CHAT
             + ["'s", "'S", "'ll", "'LL", "'re", "'Ve", "'d", "'M", "'t", " '", "'ſ"]
             + ["0", "7", "42", "123", "2024", "99999", "3.14", "1e10", "-5", "+7", "٣", "५", "²"]
             + [" ", "  ", "   ", "\t", "\n", "\r\n", "\r", "\n\n", " \n", " ", "　", " ", "\u0085"]
             + ["!", "?", ".", ",", ";", ":", "-", "_", "(", ")", "\"", "/", "\\", "|", "#", "*", "~", "`"]
             + ["a", "Z", "é", "日", "ж", "ع", "क", "ก", "😀", "👍🏽", "́", "‍", "﻿", "ſ", "ß"])

ALPHABET = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \t\n\r'\".,;:!?-_()[]{}<>|/\\@#$%^&*+=~`"
            "éèêëàâäôöûüçñßøåæœ日本語中文한국어Русскийالعربيةहिन्दीไทย😀👍🏽🏳️‍🌈́̈‍ 　 "
            "\u0085٠١٢०१२①½²ſK�\U0001f600\U00010400\U000105c0\U00011de0")


def random_strings(n: int, seed: int = 1234) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(1, 8)):
            r = rng.random()
            if r < 0.45:
                f = rng.choice(FRAGMENTS)
                if len(f) > 40 and rng.random() < 0.5:
                    a = rng.randrange(len(f))
                    f = f[a:a + rng.randint(1, 40)]
                parts.append(f)
            elif r < 0.75:
                parts.append("".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 12))))
            elif r < 0.85:
                parts.append(str(rng.randint(0, 10 ** rng.randint(1, 15))))
            elif r < 0.93:
                parts.append(rng.choice([" ", "  ", "\t", "\n", "\r\n", " \n", "\n ", " ", "　"]) *
                             rng.randint(1, 4))
            else:
                parts.append("".join(chr(rng.choice([rng.randint(0x20, 0x2FFF), rng.randint(0x3000, 0xD7FF),
                                                      rng.randint(0xE000, 0xFFFD), rng.randint(0x10000, 0x1FFFF),
                                                      rng.randint(0x20000, 0x10FFFF)]))
                                     for _ in range(rng.randint(1, 6))))
        out.append("".join(parts))
    return out


def unicode_sweep(block: int = 4096) -> list[str]:
    """Every codepoint in contexts that separate letters, numbers, spaces and the rest."""
    out = []
    for start in range(0, 0x110000, block):
        cs = [chr(c) for c in range(start, min(start + block, 0x110000)) if not 0xD800 <= c <= 0xDFFF]
        if cs:
            out.append("".join(c + "!a" + c + "a " for c in cs))
    return out


def long_documents(strings: list[str], n: int = 8, size: int = 60000, seed: int = 99) -> list[str]:
    """Long texts (exercise the piece cache and long whitespace / letter runs)."""
    rng = random.Random(seed)
    docs = []
    for _ in range(n):
        parts, total = [], 0
        while total < size:
            s = rng.choice(strings)
            parts.append(s)
            parts.append(rng.choice([" ", "\n", "\n\n", "", "  ", "\t"]))
            total += len(s) + 1
        docs.append("".join(parts))
    docs.append("a" * 20000 + " " * 20000 + "\n" * 5000 + "1" * 5000 + "!" * 5000 + "日" * 5000)
    docs.append(" " * 70000 + "x")
    docs.append("ab " * 30000)
    return docs


def adversarial_strings(seed: int = 5) -> list[str]:
    """Regex corner cases: whitespace runs before every character class, CR/LF mixtures,
    contractions after every character that could case-fold to a contraction letter, and
    partial or adjacent added tokens."""
    out = []
    tails = ["", "a", "1", "!", "\n", " x", "'s", "日"]
    for ch in [" ", "\t", "\n", "\r", "\r\n", " \n", "\n ", "　", "\u0085", " ", " ", "\x0b", "​",
               " \t", "\t\n\r "]:
        for n in (1, 2, 3, 300):
            for t in tails:
                out += [ch * n + t, t + ch * n, "w" + ch * n + t]
    rng = random.Random(seed)
    pool = ["\r", "\n", " ", "\t", "　", "\u0085", " ", "a", "1", "!", "'", "é"]
    out += ["".join(rng.choice(pool) for _ in range(rng.randint(1, 25))) for _ in range(600)]
    # Oniguruma folds only ASCII and U+017F onto the contraction letters; sweep every
    # codepoint near Latin letters (compatibility forms, ligatures, letterlike symbols too)
    near = [c for lo, hi in ((0x41, 0x24F), (0x1E00, 0x1EFF), (0x2100, 0x218F), (0xFB00, 0xFB06),
                             (0xFF21, 0xFF5A), (0x1D400, 0x1D433)) for c in range(lo, hi + 1)]
    for k in range(0, len(near), 256):
        cs = [chr(c) for c in near[k:k + 256]]
        out.append(" ".join(f"'{c}x it'{c} I'R{c} 'l{c} '{c}e" for c in cs))
    for q in ("'", "’", "`"):
        for s in ("s", "S", "ll", "LL", "lL", "re", "RE", "ve", "Ve", "m", "d", "t", "ſ", ""):
            out += [q + s, "it" + q + s, "it" + q + s + "x", " " + q + s + " ", "\n" + q + s]
    out += ["<|im_sta", "<|im_start", "<|im_start|", "<|im_start|<|im_start|>>", "<<|im_end|>>", "<|im_end|>" * 3,
            "'<|im_end|>s", "<|im_end|>'s", "1<|im_end|>1", "<|endoftext|", "<think><think>", "<tool_call", "<s", "</s",
            "<s></s>", "x<s>y</s>z", "<0x0A>", "<|im_start|>" * 40, "<|im_st" * 40, "<|im_start|>　",
            "　<|im_end|>", "<|im_end|>   x", "x   <|im_end|>", "<|<|", "▁<s>", "<s>▁"]
    # a long text (piece cache on) whose first piece has no tokens at all in SmolLM's vocabulary
    out.append("\x1d\x04 " + "ab " * 3000)
    return out


def build_corpus() -> dict[str, list[str]]:
    base = PROSE + CODE + MULTILINGUAL + CHAT
    rnd = random_strings(N_RANDOM)
    return {"curated": base, "random": rnd, "unicode": unicode_sweep(), "long": long_documents(base + rnd),
            "adversarial": adversarial_strings()}


# ---------------------------------------------------------------------------
# Hugging Face reference

class HFRef:
    def __init__(self, tj_path: str, header: dict):
        self.tok = tokenizers.Tokenizer.from_file(tj_path)
        self.meta = header["tokenizer"]
        with open(tj_path, encoding="utf-8") as f:
            raw = json.load(f)
        self.special_ids = {a["id"] for a in raw.get("added_tokens", []) if a.get("special")}
        self.nonspecial_added = [a["content"] for a in raw.get("added_tokens", []) if not a.get("special")]
        self.has_bos_template = bool(self.meta.get("add_bos"))

    def prepare(self, s: str) -> str:
        """The text borg is given: callers apply the NFC normalizer themselves."""
        if self.meta.get("nfc"):
            return self.tok.normalizer.normalize_str(s)
        return s

    def encode(self, strings: list[str], add_special_tokens: bool) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(strings, add_special_tokens=add_special_tokens)]

    def encode_no_added(self, s: str) -> list[int]:
        """normalizer -> pre-tokenizer -> BPE, with no added-token matching at all."""
        t = self.tok
        if t.normalizer is not None:
            s = t.normalizer.normalize_str(s)
        pieces = t.pre_tokenizer.pre_tokenize_str(s) if t.pre_tokenizer is not None else [(s, None)]
        ids = []
        for p, _ in pieces:
            if p:
                ids.extend(x.id for x in t.model.tokenize(p))
        return ids


# ---------------------------------------------------------------------------
# fixtures

MODELS = discover()
_CORPUS: dict[str, list[str]] | None = None


def corpus() -> dict[str, list[str]]:
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = build_corpus()
    return _CORPUS


@pytest.fixture(scope="session")
def borg():
    if not os.path.exists(LIB_PATH):
        pytest.skip(f"engine library not found at {LIB_PATH} (set BORG_LIB)")
    return Borg(LIB_PATH)


@pytest.fixture(scope="module", params=MODELS, ids=[m[0] for m in MODELS])
def model(request, borg):
    src, path, tj = request.param
    header = read_header(path)
    m = borg.load(path)
    yield src, m, HFRef(tj, header)
    borg.lib.borg_model_free(m)


def _describe(ref: HFRef, s: str, got: list[int], want: list[int]) -> str:
    k = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
    tok = ref.tok
    return (f"{s[:120]!r}: first difference at {k}: borg {got[k:k + 6]} "
            f"{[tok.id_to_token(i) for i in got[k:k + 6]]} vs hf {want[k:k + 6]} "
            f"{[tok.id_to_token(i) for i in want[k:k + 6]]}")


def compare_encode(borg: Borg, m: int, ref: HFRef, strings: list[str], add_bos: int, special: int,
                   add_special_tokens: bool) -> list[str]:
    prepared = [ref.prepare(s) for s in strings]
    want = ref.encode(prepared, add_special_tokens)
    bad = []
    for s, w in zip(prepared, want):
        got = borg.tokenize(m, s.encode("utf-8"), add_bos, special)
        if got != w:
            bad.append(_describe(ref, s, got, w))
    return bad


# ---------------------------------------------------------------------------
# tests

pytestmark = pytest.mark.skipif(not MODELS, reason="no .borg models with a matching tokenizer.json "
                                                   "(set BORG_MODELS_DIR and HF_MODELS_DIR)")


@pytest.mark.parametrize("part", ["curated", "random", "unicode", "long", "adversarial"])
def test_encode_parity(borg, model, part):
    _, m, ref = model
    bad = compare_encode(borg, m, ref, corpus()[part], 0, 1, False)
    assert not bad, f"{len(bad)}/{len(corpus()[part])} mismatches, e.g.\n" + "\n".join(bad[:5])


@pytest.mark.parametrize("part", ["curated", "random"])
def test_add_special_tokens(borg, model, part):
    _, m, ref = model
    strings = corpus()[part]
    bad = compare_encode(borg, m, ref, strings, -1, 1, True)  # model default == HF add_special_tokens=True
    if ref.has_bos_template:
        bad += compare_encode(borg, m, ref, strings, 1, 1, True)
    assert not bad, f"{len(bad)} mismatches, e.g.\n" + "\n".join(bad[:5])


@pytest.mark.parametrize("part", ["curated", "random", "long", "adversarial"])
def test_no_special_parsing(borg, model, part):
    """parse_special=0: added tokens are never matched; text goes through normalizer/pre-tokenizer/BPE."""
    _, m, ref = model
    strings = [ref.prepare(s) for s in corpus()[part]]
    bad = []
    for s in strings:
        got = borg.tokenize(m, s.encode("utf-8"), 0, 0)
        want = ref.encode_no_added(s)
        if got != want:
            bad.append(_describe(ref, s, got, want))
    # HF's split_special_tokens agrees as long as no non-special added token occurs in the text
    ref.tok.encode_special_tokens = True
    try:
        plain = [s for s in strings if not any(a in s for a in ref.nonspecial_added)]
        for s, w in zip(plain, ref.encode(plain, False)):
            got = borg.tokenize(m, s.encode("utf-8"), 0, 0)
            if got != w:
                bad.append("split_special_tokens: " + _describe(ref, s, got, w))
    finally:
        ref.tok.encode_special_tokens = False
    assert not bad, f"{len(bad)} mismatches, e.g.\n" + "\n".join(bad[:5])


def _dropped_bytes(borg: Borg, m: int) -> set[int]:
    """Bytes without a token in a byte-level vocabulary: HF drops them silently."""
    return {b for b in range(256) if not borg.tokenize(m, bytes([b]), 0, 0)}


@pytest.mark.parametrize("part", ["curated", "random", "unicode", "long", "adversarial"])
def test_roundtrip(borg, model, part):
    _, m, ref = model
    bad = []
    if ref.meta["byte_level"]:
        dropped = _dropped_bytes(borg, m)
        for s in corpus()[part]:
            data = ref.prepare(s).encode("utf-8")
            want = bytes(b for b in data if b not in dropped)
            ids = borg.tokenize(m, data, 0, 1)
            if borg.detokenize(m, ids) != want:
                bad.append(repr(s[:80]))
            ids = borg.tokenize(m, data, 0, 0)
            if borg.detokenize(m, ids) != want:
                bad.append("no-special " + repr(s[:80]))
    else:
        # one segment (no added-token matching): "▁" prefix + spaces as "▁" decode back exactly,
        # unless the text itself contains "▁"
        for s in corpus()[part]:
            if "▁" in s:
                continue
            ids = borg.tokenize(m, s.encode("utf-8"), 0, 0)
            if borg.detokenize(m, ids) != s.encode("utf-8"):
                bad.append(repr(s[:80]))
    assert not bad, f"{len(bad)} round-trip failures, e.g.\n" + "\n".join(bad[:5])


@pytest.mark.parametrize("part", ["curated", "random"])
def test_decode_parity(borg, model, part):
    """borg_detokenize == Tokenizer.decode for HF's own encodings (where the text is valid UTF-8)."""
    _, m, ref = model
    bad = []
    prepared = [ref.prepare(s) for s in corpus()[part]]
    for ids in ref.encode(prepared, True):
        for skip in (False, True):
            got = borg.detokenize(m, ids, int(skip))
            try:
                got_s = got.decode("utf-8")
            except UnicodeDecodeError:
                continue
            want = ref.tok.decode(ids, skip_special_tokens=skip)
            if got_s != want:
                bad.append(f"skip={skip} {ids[:12]}: borg {got_s[:60]!r} hf {want[:60]!r}")
    assert not bad, f"{len(bad)} decode mismatches, e.g.\n" + "\n".join(bad[:5])


def test_vocab_tables(borg, model):
    """borg_token_bytes / borg_token_type / borg_is_eog against tokenizer.json."""
    _, m, ref = model
    tok = ref.tok
    byte_level = ref.meta["byte_level"]
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, k = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + k)
            k += 1
    u2b = {chr(c): b for b, c in zip(bs, cs)}
    n_vocab = ref.meta["n_vocab"]
    bad = []
    for i in range(n_vocab):
        s = tok.id_to_token(i)
        if s is None:
            continue
        if i in ref.special_ids or s in ref.nonspecial_added:
            want = s.encode("utf-8")
        elif byte_level:
            want = bytes(u2b[c] for c in s) if all(c in u2b for c in s) else s.encode("utf-8")
        elif len(s) == 6 and s.startswith("<0x") and s.endswith(">"):
            want = bytes([int(s[3:5], 16)])
        else:
            want = s.replace("▁", " ").encode("utf-8")
        if borg.token_bytes(m, i) != want:
            bad.append(f"{i} {s!r}: {borg.token_bytes(m, i)!r} != {want!r}")
        t = borg.lib.borg_token_type(m, i)
        if (t == 2) != (i in ref.special_ids):
            bad.append(f"{i} {s!r}: type {t}")
    assert borg.token_bytes(m, -1) == b"" and borg.token_bytes(m, n_vocab) == b""
    assert borg.lib.borg_token_type(m, n_vocab) == 4
    for e in ref.meta["eos_ids"]:
        assert borg.lib.borg_is_eog(m, e)
    assert not borg.lib.borg_is_eog(m, -1) and not borg.lib.borg_is_eog(m, n_vocab + 5)
    assert not bad, f"{len(bad)} vocab mismatches, e.g.\n" + "\n".join(bad[:5])


def test_buffer_and_edge_cases(borg, model):
    _, m, ref = model
    text = "Hello world, this is a buffer test with several tokens."
    data = text.encode()
    full = borg.tokenize(m, data, 0, 1)
    n = len(full)
    buf = (ctypes.c_int32 * n)()
    assert borg.lib.borg_tokenize(m, data, len(data), 0, 1, buf, n) == n
    assert borg.lib.borg_tokenize(m, data, len(data), 0, 1, buf, n - 1) == -n
    assert borg.lib.borg_tokenize(m, data, len(data), 0, 1, None, 0) == -n
    assert borg.lib.borg_tokenize(m, data, -1, 0, 1, buf, n) == n  # NUL-terminated
    assert borg.lib.borg_tokenize(m, b"", 0, 0, 1, buf, n) == 0
    assert borg.lib.borg_tokenize(m, None, 0, 0, 1, buf, n) == 0
    bos = ref.meta["bos_id"]
    with_bos = borg.tokenize(m, data, 1, 1)
    assert with_bos == ([bos] + full if bos >= 0 else full)
    assert borg.tokenize(m, b"", 1, 1) == ([bos] if bos >= 0 else [])
    # detokenize reports the full length even when the buffer is short
    out = borg.detokenize(m, full)
    small = ctypes.create_string_buffer(4)
    arr = (ctypes.c_int32 * n)(*full)
    assert borg.lib.borg_detokenize(m, arr, n, 0, small, 4) == len(out)
    assert small.raw[:4] == out[:4]


def test_invalid_utf8(borg, model):
    """Arbitrary bytes never crash; byte-level models keep every byte that has a token."""
    _, m, ref = model
    rng = random.Random(7)
    dropped = _dropped_bytes(borg, m) if ref.meta["byte_level"] else set()
    samples = [bytes(rng.randrange(256) for _ in range(rng.randint(1, 64))) for _ in range(2000)]
    samples += [b"\xff", b"\xc0\x80", b"\xed\xa0\x80", b"\xf4\x90\x80\x80", b"abc\xe2\x96", b"\xe2\x96\x81\xe2",
                b"'\xc5", b"' \xff", "日本".encode()[:-1], b"\xf0\x9f\x98", b"<|im_start|\xff>"]
    for data in samples:
        for special in (0, 1):
            ids = borg.tokenize(m, data, 0, special)
            back = borg.detokenize(m, ids)
            if ref.meta["byte_level"]:
                assert back == bytes(b for b in data if b not in dropped), data


def test_throughput(borg, model):
    """~1 MB of mixed text must tokenize well under a second (generous bound for busy machines)."""
    _, m, ref = model
    text = ("\n".join(corpus()["curated"]) + "\n").encode()
    data = (text * (1_000_000 // len(text) + 1))[:1_000_000]
    t0 = time.perf_counter()
    ids = borg.tokenize(m, data, 0, 1)
    dt = time.perf_counter() - t0
    assert ids
    assert dt < 3.0, f"{dt:.2f} s for 1 MB"


def _colliding_words(prefix: bytes, n: int, window: int = 1024, seed: int = 0) -> list[bytes]:
    """n distinct 8-letter words whose cache key (prefix + word) has an FNV-1a hash in the
    first `window` slots of tokenizer.c's piece cache (2^18 slots for texts >= 1 MB)."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed)
    pre = np.frombuffer(prefix, dtype=np.uint8)
    found: dict[bytes, None] = {}
    while len(found) < n:
        cand = rng.integers(ord("a"), ord("z") + 1, size=(1 << 21, 8), dtype=np.uint8)
        h = np.full(len(cand), 2166136261, dtype=np.uint32)
        for col in [np.full(len(cand), b, dtype=np.uint8) for b in pre] + [cand[:, j] for j in range(8)]:
            h ^= col.astype(np.uint32)
            h *= np.uint32(16777619)
        for w in cand[(h & ((1 << 18) - 1)) < window][: n - len(found)]:
            found[w.tobytes()] = None
    return list(found)


def test_cache_collisions(borg, model):
    """Words crafted to share one probe cluster of the piece cache must not make long texts
    quadratic (unbounded linear probing took ~40 s on 5 MB of them)."""
    _, m, ref = model
    prefix = b" " if ref.meta["byte_level"] else "▁".encode()
    size = 2_000_000

    def timed(words: list[bytes]) -> float:
        text = b"".join(b" " + w for w in words)
        data = (text * (size // len(text) + 1))[:size]
        t0 = time.perf_counter()
        n = borg.lib.borg_tokenize(m, data, len(data), 0, 1, None, 0)
        assert n < 0
        return time.perf_counter() - t0

    rng = random.Random(0)
    control = [bytes(rng.randrange(97, 123) for _ in range(8)) for _ in range(32768)]
    t_control = timed(control)
    t_attack = timed(_colliding_words(prefix, 32768))
    assert t_attack < 1.0 + 10 * t_control, f"{t_attack:.2f} s for colliding words vs {t_control:.2f} s"


# ---------------------------------------------------------------------------
# derived tokenizers: features none of the test models use (llama3 / plain gpt2 regexes,
# ignore_merges, SentencePiece without byte fallback or prefix), built by editing a real
# tokenizer.json and converting it together with a tiny random-free model

LLAMA3_REGEX = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                r"|\s*[\r\n]+|\s+(?!\S)|\s+")
GPT2_REGEX = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


def _byte_level_str(text: str) -> str:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, k = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + k)
            k += 1
    m = dict(zip(bs, cs))
    return "".join(chr(m[b]) for b in text.encode("utf-8"))


def _add_tokens(tj, pieces: list[str], merges: list[tuple[str, str]] = ()) -> None:
    """Append vocabulary entries (and lowest-priority merges) to a BPE tokenizer.json. HF numbers
    added tokens that are not in the model vocabulary from the vocabulary size on, so the new
    entries take the ids right after the vocabulary and such added tokens move up."""
    vocab = tj["model"]["vocab"]
    assert sorted(vocab.values()) == list(range(len(vocab)))
    new = [p for p in dict.fromkeys(pieces) if p not in vocab]
    base = len(vocab)
    for a in tj.get("added_tokens", []):
        if a["content"] not in vocab and a["id"] >= base:
            a["id"] += len(new)
    for i, p in enumerate(new):
        vocab[p] = base + i
    ml = tj["model"]["merges"]
    for a, b in merges:
        ml.append([a, b] if ml and isinstance(ml[0], list) else f"{a} {b}")


# extra text for the derived tokenizers' new entries
VARIANT_TEXT = ["xyzzy plugh", " xyzzy", "plugh xyzzy", "xyzzy", "1234567 12 999 0 00 000 0000 31415926",
                "a1b22c333d4444", "2024-12-31", "x xyzzyx xyzzy, xyzzy.", "plughxyzzy"]


def _variant_llama3(tj):
    tj["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = LLAMA3_REGEX
    tj["model"]["ignore_merges"] = True
    digits = "0123456789"
    two = [a + b for a in digits for b in digits]
    three = [a + b + c for a in digits for b in digits for c in digits]
    # multi-digit tokens reached by merges, and whole words that only ignore_merges can produce
    _add_tokens(tj, two + three + [_byte_level_str(w) for w in ("xyzzy", " xyzzy", "plugh")],
                [(x[0], x[1]) for x in two] + [(x[:2], x[2]) for x in three])


def _variant_gpt2_split(tj):
    tj["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = GPT2_REGEX


def _variant_gpt2_bytelevel(tj):
    tj["pre_tokenizer"] = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}


def _variant_spm_unk(tj):
    tj["model"]["byte_fallback"] = False


def _variant_spm_noprefix(tj):
    tj["normalizer"] = {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}


def _variant_spm_partial_bytes(tj):
    # characters whose bytes lack a byte token fall back to <unk>, the others to byte tokens
    for b in ("<0xE6>", "<0x80>", "<0xF0>"):
        del tj["model"]["vocab"][b]


def _variant_spm_ignore_merges(tj):
    # "▁plugh▁xyzzy" holds "▁" after a letter, which also turns off the cut-before-"▁" shortcut
    tj["model"]["ignore_merges"] = True
    _add_tokens(tj, ["▁xyzzy", "▁plugh▁xyzzy"])


# name -> (base tokenizer kind, edit); "qwen2" / "smollm" bases are byte-level, "spm" is SentencePiece
VARIANTS = {
    "llama3-regex+ignore_merges": ("qwen2", _variant_llama3),
    "gpt2-regex-split": ("qwen2", _variant_gpt2_split),
    "gpt2-bytelevel": ("smollm", _variant_gpt2_bytelevel),
    "spm-unk-no-byte-fallback": ("spm", _variant_spm_unk),
    "spm-no-prefix": ("spm", _variant_spm_noprefix),
    "spm-ignore-merges": ("spm", _variant_spm_ignore_merges),
    "spm-partial-byte-fallback": ("spm", _variant_spm_partial_bytes),
}


def _base_for(kind: str):
    for src, path, tj in MODELS:
        meta = read_header(path)["tokenizer"]
        k = "spm" if not meta["byte_level"] else meta["pretokenizer"]
        if k == kind:
            return src, path, tj
    return None


def _write_safetensors(path: str, tensors: dict) -> None:
    import numpy as np
    header, blobs, off = {}, [], 0
    for name, a in tensors.items():
        b = np.ascontiguousarray(a, dtype=np.float32).tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for b in blobs:
            f.write(b)


def build_variant(name: str, out_dir: str) -> tuple[str, str] | None:
    """Write <out_dir>/<name>/ (HF folder) and <out_dir>/<name>.borg; returns their paths."""
    np = pytest.importorskip("numpy")
    sys.path.insert(0, ROOT)
    from superborg.convert import convert
    kind, edit = VARIANTS[name]
    base = _base_for(kind)
    if base is None:
        return None
    hf_dir = os.path.join(out_dir, name)
    os.makedirs(hf_dir, exist_ok=True)
    with open(base[2], encoding="utf-8") as f:
        tj = json.load(f)
    edit(tj)
    with open(os.path.join(hf_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tj, f, ensure_ascii=False)
    base_dir = os.path.dirname(base[2])
    tc = os.path.join(base_dir, "tokenizer_config.json")
    if os.path.exists(tc):
        with open(tc, encoding="utf-8") as f, open(os.path.join(hf_dir, "tokenizer_config.json"), "w") as g:
            g.write(f.read())
    with open(os.path.join(base_dir, "config.json")) as f:
        bcfg = json.load(f)
    ids = list(tj["model"]["vocab"].values()) + [a["id"] for a in tj.get("added_tokens", [])]
    vocab = max(ids) + 1
    d = 32
    cfg = {"model_type": "llama", "hidden_size": d, "num_hidden_layers": 1, "num_attention_heads": 1,
           "num_key_value_heads": 1, "intermediate_size": d, "vocab_size": vocab, "max_position_embeddings": 64,
           "rms_norm_eps": 1e-6, "tie_word_embeddings": True,
           "bos_token_id": bcfg.get("bos_token_id"), "eos_token_id": bcfg.get("eos_token_id")}
    with open(os.path.join(hf_dir, "config.json"), "w") as f:
        json.dump(cfg, f)
    z = lambda *shape: np.zeros(shape, dtype=np.float32)  # noqa: E731
    p = "model.layers.0."
    _write_safetensors(os.path.join(hf_dir, "model.safetensors"), {
        "model.embed_tokens.weight": z(vocab, d), "model.norm.weight": z(d),
        p + "input_layernorm.weight": z(d), p + "post_attention_layernorm.weight": z(d),
        p + "self_attn.q_proj.weight": z(d, d), p + "self_attn.k_proj.weight": z(d, d),
        p + "self_attn.v_proj.weight": z(d, d), p + "self_attn.o_proj.weight": z(d, d),
        p + "mlp.gate_proj.weight": z(d, d), p + "mlp.up_proj.weight": z(d, d), p + "mlp.down_proj.weight": z(d, d),
    })
    out = os.path.join(out_dir, name + ".borg")
    convert(hf_dir, out, "f32", source=name, log=None)
    return out, os.path.join(hf_dir, "tokenizer.json")


@pytest.fixture(scope="module", params=list(VARIANTS))
def variant(request, borg, tmp_path_factory):
    built = build_variant(request.param, str(tmp_path_factory.mktemp("tokvariants")))
    if built is None:
        pytest.skip(f"no base tokenizer of kind {VARIANTS[request.param][0]}")
    path, tj = built
    m = borg.load(path)
    yield request.param, m, HFRef(tj, read_header(path))
    borg.lib.borg_model_free(m)


def test_variant_parity(borg, variant):
    name, m, ref = variant
    c = corpus()
    strings = VARIANT_TEXT + c["curated"] + c["random"][:1500] + c["long"][:3]
    if ref.meta["byte_level"]:
        strings += c["unicode"]
    expect = {"llama3-regex+ignore_merges": ("llama3", True), "gpt2-regex-split": ("gpt2", True),
              "gpt2-bytelevel": ("gpt2", True), "spm-unk-no-byte-fallback": ("none", False),
              "spm-no-prefix": ("none", False), "spm-ignore-merges": ("none", False),
              "spm-partial-byte-fallback": ("none", False)}[name]
    assert (ref.meta["pretokenizer"], ref.meta["byte_level"]) == expect
    bad = compare_encode(borg, m, ref, strings, 0, 1, False)
    bad += compare_encode(borg, m, ref, c["curated"], -1, 1, True)
    for s in strings[:2000]:
        s = ref.prepare(s)
        if borg.tokenize(m, s.encode("utf-8"), 0, 0) != ref.encode_no_added(s):
            bad.append("no-special " + repr(s[:80]))
    assert not bad, f"{len(bad)} mismatches, e.g.\n" + "\n".join(bad[:5])


# ---------------------------------------------------------------------------
# script mode: parity report

def _report() -> int:
    if not os.path.exists(LIB_PATH) or not MODELS:
        print("set BORG_LIB, BORG_MODELS_DIR and HF_MODELS_DIR", file=sys.stderr)
        return 1
    b = Borg(LIB_PATH)
    parts = corpus()
    bench = os.environ.get("BORG_BENCH_TEXT")
    for src, path, tj in MODELS:
        header = read_header(path)
        ref = HFRef(tj, header)
        m = b.load(path)
        print(f"== {src} ({os.path.basename(path)})")
        for name, strings in parts.items():
            for add_bos, special, ast in ((0, 1, False), (-1, 1, True)):
                bad = compare_encode(b, m, ref, strings, add_bos, special, ast)
                print(f"  {name:8s} add_bos={add_bos:2d}: {len(strings) - len(bad)}/{len(strings)} identical")
                for x in bad[:3]:
                    print("     ", x)
            bad = sum(1 for s in strings if b.tokenize(m, ref.prepare(s).encode(), 0, 0)
                      != ref.encode_no_added(ref.prepare(s)))
            print(f"  {name:8s} parse_special=0: {len(strings) - bad}/{len(strings)} identical")
        data = open(bench, "rb").read() if bench else ("\n".join(parts["curated"]) * 200).encode()
        for special in (1, 0):
            t0 = time.perf_counter()
            ids = b.tokenize(m, data, 0, special)
            dt = time.perf_counter() - t0
            print(f"  throughput: {len(data) / 1e6:.2f} MB -> {len(ids)} tokens in {dt * 1000:.0f} ms "
                  f"({len(data) / 1e6 / dt:.1f} MB/s, parse_special={special})")
        t0 = time.perf_counter()
        hf_ids = ref.tok.encode(ref.prepare(data.decode("utf-8", errors="replace")), add_special_tokens=False).ids
        dt_hf = time.perf_counter() - t0
        print(f"  hf tokenizers same text: {dt_hf * 1000:.0f} ms ({len(hf_ids)} tokens)")
        b.lib.borg_model_free(m)
    return 0


if __name__ == "__main__":
    sys.exit(_report())
