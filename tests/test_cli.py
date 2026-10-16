"""The command-line interface, run as `python -m superborg ...` in subprocesses.

Uses smollm2-135m-q8_0.borg (and its q4_0 sibling as a draft model); tests skip when the
files are missing. The convert test needs the Hugging Face checkpoint of SmolLM2-135M
(HF_MODELS_DIR); the network test of `pull` runs only with SUPERBORG_TEST_PULL=1.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
THREADS = os.environ.get("BORG_TEST_THREADS", "4")
MODEL = "smollm2-135m-q8_0.borg"


def _dirs(env: str) -> list[str]:
    dirs = [os.environ.get(env, ""), os.path.join(ROOT, "models"), os.path.expanduser("~/.cache/superborg")]
    try:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import hf_oracle  # noqa: PLC0415 - optional: this checkout's default model directories

        dirs.append(getattr(hf_oracle, env))
    except Exception:  # noqa: BLE001
        pass
    return [d for d in dirs if d and os.path.isdir(d)]


def model_path(name: str) -> str:
    for d in _dirs("BORG_MODELS_DIR"):
        if os.path.isfile(os.path.join(d, name)):
            return os.path.join(d, name)
    pytest.skip(f"{name} not found (set BORG_MODELS_DIR)")


def hf_path(name: str) -> str:
    for d in _dirs("HF_MODELS_DIR"):
        if os.path.isfile(os.path.join(d, name, "config.json")):
            return os.path.join(d, name)
    pytest.skip(f"Hugging Face checkpoint {name} not found (set HF_MODELS_DIR)")


def sb(*args: str, stdin: str | None = None, env: dict | None = None, timeout: float = 600,
       check: bool = True) -> subprocess.CompletedProcess:
    full_env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""), **(env or {}))
    p = subprocess.run([sys.executable, "-m", "superborg", *args], input=stdin, capture_output=True, text=True,
                       cwd=ROOT, env=full_env, timeout=timeout)
    if check:
        assert p.returncode == 0, f"exit {p.returncode}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"
    return p


@pytest.fixture(scope="module")
def model():
    return model_path(MODEL)


# ---------------------------------------------------------------------------

def test_help():
    out = sb("--help").stdout
    for cmd in ("pull", "convert", "info", "run", "chat", "ask", "serve", "bench"):
        assert cmd in out
        assert sb(cmd, "--help").stdout.startswith("usage: superborg " + cmd)
    p = sb(check=False)
    assert p.returncode == 2 and "usage" in p.stderr


def test_missing_model():
    p = sb("info", "/no/such/model.borg", check=False)
    assert p.returncode == 2 and "model not found" in p.stderr
    p = sb("run", "does-not-exist", "-p", "hi", check=False)
    assert p.returncode == 2 and "model not found" in p.stderr


def test_argument_validation(tmp_path):
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text("{}")
    offline = {"HF_ENDPOINT": "http://127.0.0.1:9"}  # nothing listens there
    # rejected by argparse before anything is loaded or downloaded
    for args in (("convert", str(hf), str(tmp_path / "x.borg"), "--qtype", "q5"),
                 ("pull", "org/repo", "--dest", str(tmp_path), "--qtype", "int3"),
                 ("bench", "m.borg", "--repeat", "0"),
                 ("bench", "m.borg", "--prompt", "0"),
                 ("ask", "m.borg", "q", "--samples", "0")):
        p = sb(*args, env=offline, check=False)
        assert p.returncode == 2 and "error: argument" in p.stderr and "Traceback" not in p.stderr, (args, p.stderr)
    # a missing calibration file is reported before `pull` touches the network
    p = sb("pull", "org/repo", "--dest", str(tmp_path), "--qtype", "q4_0", "--calibrate", str(tmp_path / "nope.txt"),
           env={"HF_ENDPOINT": "http://127.0.0.1:9"}, check=False)
    assert p.returncode == 1 and "nope.txt" in p.stderr and "Traceback" not in p.stderr, p.stderr


def test_os_errors_are_reported(model, tmp_path):
    p = sb("run", model, "-f", str(tmp_path), "-t", THREADS, check=False)  # a directory
    assert p.returncode == 1 and "IsADirectoryError" in p.stderr and "Traceback" not in p.stderr, p.stderr
    # network failures of `pull` (nothing listens on port 9)
    p = sb("pull", "org/repo", "--dest", str(tmp_path), env={"HF_ENDPOINT": "http://127.0.0.1:9"}, check=False)
    assert p.returncode == 1 and "Traceback" not in p.stderr, p.stderr


def test_ambiguous_model_name(model, tmp_path):
    for q in ("q8_0", "q4_0"):
        os.symlink(model, tmp_path / f"smol-{q}.borg")
    env = {"SUPERBORG_MODELS": str(tmp_path)}
    p = sb("info", "smol", env=env, check=False)
    assert p.returncode == 2 and "ambiguous" in p.stderr and "smol-q4_0.borg" in p.stderr
    assert json.loads(sb("info", "smol-q8", "--json", env=env).stdout)["quant"] == "q8_0"


def test_info(model):
    out = sb("info", model).stdout
    assert "llama" in out and "30 layers" in out and "chat format" in out and "simd" in out
    info = json.loads(sb("info", model, "--json").stdout)
    assert info["arch"] == "llama" and info["quant"] == "q8_0"
    assert 130e6 < info["parameters"] < 140e6
    assert info["tokenizer"]["n_vocab"] == 49152
    assert info["chat"]["format"] == "chatml" and "<|im_end|>" in info["chat"]["stop_strings"]
    assert info["engine"]["simd"]


def test_run_greedy_matches_library(model):
    from superborg import LLM
    with LLM(model, threads=int(THREADS), n_ctx=256) as llm:
        expect = llm.complete("The capital of France is", max_tokens=12, temperature=0.0).text
    p = sb("run", model, "-p", "The capital of France is", "-n", "12", "-t", THREADS, "-v")
    assert p.stdout == expect + "\n"
    assert "tok/s" in p.stderr and "stop:" in p.stderr
    # the prompt can come from stdin or a file
    assert sb("run", model, "-p", "-", "-n", "12", "-t", THREADS, stdin="The capital of France is").stdout == p.stdout
    assert sb("run", model, "--echo", "-p", "The capital of France is", "-n", "12", "-t", THREADS).stdout \
        == "The capital of France is" + expect + "\n"


def test_run_options(model, tmp_path):
    f = tmp_path / "prompt.txt"
    f.write_text("List three animals:")
    p = sb("run", model, "-f", str(f), "--chat", "--system", "Be brief.", "-n", "16", "-t", THREADS,
           "--temp", "0.8", "--top-k", "40", "--top-p", "0.9", "--min-p", "0.05", "--repeat-penalty", "1.1",
           "--seed", "7", "--kv-f16", "-c", "512")
    assert p.stdout.strip()
    again = sb("run", model, "-f", str(f), "--chat", "--system", "Be brief.", "-n", "16", "-t", THREADS,
               "--temp", "0.8", "--top-k", "40", "--top-p", "0.9", "--min-p", "0.05", "--repeat-penalty", "1.1",
               "--seed", "7", "--kv-f16", "-c", "512")
    assert again.stdout == p.stdout, "a fixed seed reproduces the output"
    ref = sb("run", model, "-p", "1, 2, 3, 4, 5,", "-n", "16", "-t", THREADS).stdout
    stop = ref[4:7]
    out = sb("run", model, "-p", "1, 2, 3, 4, 5,", "-n", "16", "-t", THREADS, "--stop", stop, "-v")
    assert out.stdout == ref[:ref.find(stop)] + "\n" and "stop: stop" in out.stderr
    p = sb("run", model, check=False)
    assert p.returncode == 2 and "prompt" in p.stderr


def test_run_speculative(model):
    text = "alpha beta gamma delta epsilon zeta alpha beta gamma delta epsilon zeta alpha beta"
    p = sb("run", model, "-p", text, "-n", "16", "-t", THREADS, "--lookup", "3", "-v")
    assert "speculative:" in p.stderr and p.stdout.strip()
    draft = model_path("smollm2-135m-q4_0.borg")
    p = sb("run", model, "-p", text, "-n", "16", "-t", THREADS, "--draft", draft, "--n-draft", "4", "-v")
    assert "speculative:" in p.stderr and "drafts accepted" in p.stderr


def test_chat_repl(model):
    script = "Hello!\n/stats\n/reset\n/system You answer in one word.\nWhat color is the sky?\n/help\n/bogus\n\n/exit\n"
    p = sb("chat", model, "-n", "12", "-t", THREADS, "--temp", "0", stdin=script)
    out = p.stdout
    assert "(stats on)" in out and "(conversation cleared)" in out
    assert "(system prompt set; conversation cleared)" in out
    assert "/reset" in out and "(unknown command /bogus" in out
    assert "prompt:" in p.stderr  # /stats prints timings to stderr after each reply
    # EOF ends the session like /exit
    p = sb("chat", model, "-n", "4", "-t", THREADS, stdin="Hi\n")
    assert p.stdout.strip()


def test_chat_history_is_kept(model):
    # with history, the second reply's prompt is served largely from the KV cache
    p = sb("chat", model, "-n", "8", "-t", THREADS, "--temp", "0", "-v", stdin="My name is Ada.\nWhat is my name?\n")
    lines = [ln for ln in p.stderr.splitlines() if ln.startswith("superborg: prompt:")]
    assert len(lines) == 2
    cached = int(lines[1].split(" cached")[0].rsplit(" ", 1)[1])
    assert cached > 20


def _reason_available() -> bool:
    try:
        import superborg.reason  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _reason_available(), reason="superborg.reason unavailable")
def test_ask(model, tmp_path):
    p = sb("ask", model, "What is 12 * 12 + 1?", "-v")
    assert "145" in p.stdout and "strategy: calc" in p.stderr
    d = json.loads(sb("ask", model, "What is 7 * 6?", "--json").stdout)
    assert d["answer"] == "42" and d["strategy"] == "calc" and "candidates" in d
    p = sb("ask", model, "What is the capital of France?", "--strategy", "direct", "-n", "16", "-t", THREADS)
    assert p.stdout.strip()
    p = sb("ask", model, "x", "--strategy", "telepathy", check=False)
    assert p.returncode == 2 and "unknown strategy" in p.stderr and "pal_vote" in p.stderr
    p = sb("ask", model, "x", "--strategy", "rag", check=False)
    assert p.returncode == 2 and "context" in p.stderr


@pytest.mark.skipif(not _reason_available(), reason="superborg.reason unavailable")
def test_ask_context_dir(model, tmp_path):
    (tmp_path / "notes.md").write_text("# Office\n\nThe office wifi password is tulip-77.\n\n"
                                       "Lunch is served at noon in the atrium.\n")
    (tmp_path / "other.txt").write_text("Bananas are yellow and grow in bunches.\n")
    p = sb("ask", model, "What is the office wifi password?", "--context-dir", str(tmp_path), "-n", "32",
           "-t", THREADS, "-v")
    assert "strategy: rag" in p.stderr and "notes.md" in p.stderr
    assert p.stdout.strip()
    p = sb("ask", model, "q", "--context-dir", str(tmp_path / "missing"), check=False)
    assert p.returncode == 2 and "not a directory" in p.stderr


def test_bench(model):
    d = json.loads(sb("bench", model, "--prompt", "48", "--gen", "16", "-t", THREADS, "--lookup", "3",
                      "--json").stdout)
    plain, look = d["results"]["plain"], d["results"]["lookup"]
    assert plain["prompt_tokens"] == 48 and plain["gen_tokens"] == 16
    assert plain["prefill_tok_s"] > 0 and plain["decode_tok_s"] > 0
    assert look["drafted"] > 0 and "speedup" in look and "matches_plain" in look
    out = sb("bench", model, "--prompt", "32", "--gen", "8", "-t", THREADS, "--draft",
             model_path("smollm2-135m-q4_0.borg")).stdout
    assert "plain" in out and "draft" in out and "tok/s" in out and "identical to plain" in out


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_serve(model):
    port = _free_port()
    env = dict(os.environ, PYTHONPATH=ROOT)
    proc = subprocess.Popen([sys.executable, "-m", "superborg", "serve", model, "--port", str(port), "-t", THREADS,
                             "-c", "512", "--model-id", "smol"], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(300):
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if json.load(r)["status"] == "ok":
                        break
            except OSError:
                if proc.poll() is not None:
                    pytest.fail(f"server exited: {proc.stderr.read()}")
                time.sleep(0.1)
        with urllib.request.urlopen(base + "/v1/models", timeout=10) as r:
            assert json.load(r)["data"][0]["id"] == "smol"
        req = urllib.request.Request(base + "/v1/chat/completions", method="POST",
                                     data=json.dumps({"messages": [{"role": "user", "content": "Hi"}],
                                                      "max_tokens": 6}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.load(r)
        assert body["model"] == "smol" and body["usage"]["completion_tokens"] > 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    assert "serving smol" in proc.stderr.read()


def test_convert_and_name_lookup(tmp_path):
    hf = hf_path("HuggingFaceTB_SmolLM2-135M-Instruct")
    out = tmp_path / "smol-q4_0.borg"
    p = sb("convert", hf, str(out), "--qtype", "q4_0", "--source", "HuggingFaceTB/SmolLM2-135M-Instruct")
    assert out.exists() and "wrote" in p.stderr
    info = json.loads(sb("info", str(out), "--json").stdout)
    assert info["quant"] == "q4_0" and info["source"] == "HuggingFaceTB/SmolLM2-135M-Instruct"
    # models in $SUPERBORG_MODELS are found by name
    env = {"SUPERBORG_MODELS": str(tmp_path)}
    assert json.loads(sb("info", "smol-q4_0", "--json", env=env).stdout)["quant"] == "q4_0"
    assert sb("run", "smol", "-p", "Hello", "-n", "4", "-t", THREADS, env=env).stdout.strip()
    p = sb("convert", str(tmp_path), str(tmp_path / "x.borg"), check=False)
    assert p.returncode == 2 and "config.json" in p.stderr
    out.unlink()


def test_pull_reuses_existing_conversion(model, tmp_path):
    # an existing conversion is reused without touching the network
    os.symlink(model, tmp_path / "smollm2-135m-instruct-q8_0.borg")
    p = sb("pull", "HuggingFaceTB/SmolLM2-135M-Instruct", "--dest", str(tmp_path),
           env={"HF_ENDPOINT": "http://127.0.0.1:9"})
    assert p.stdout.strip() == str(tmp_path / "smollm2-135m-instruct-q8_0.borg")
    assert "exists" in p.stderr


@pytest.mark.skipif(os.environ.get("SUPERBORG_TEST_PULL") != "1", reason="network test; set SUPERBORG_TEST_PULL=1")
def test_pull_network(tmp_path):
    p = sb("pull", "HuggingFaceTB/SmolLM2-135M-Instruct", "--dest", str(tmp_path), "--qtype", "q4_0", "-q",
           timeout=3600)
    path = p.stdout.strip()
    assert path.endswith("smollm2-135m-instruct-q4_0.borg") and os.path.isfile(path)
    assert not os.path.exists(os.path.join(tmp_path, "hf")), "HF files are removed without --keep-hf"
    assert sb("run", path, "-p", "Hi", "-n", "2", "-t", THREADS).stdout.strip()
    shutil.rmtree(tmp_path, ignore_errors=True)
