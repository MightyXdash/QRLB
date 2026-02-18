import os
import sys
import json
import time
import random
import shutil
import socket
import platform
import datetime
import urllib.request
import urllib.error
import getpass
import hashlib
import threading
from pathlib import Path

code_id = "deepseek_var_20240620_001"


model = "qwen3:4b-thinking-2507-q4_K_M"

think_level = "high"
num_ctx = 4000
num_predict = 3000


USE_CLOUD = False


LOCAL_BASE = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


CLOUD_BASE = "https://ollama.com"


OLLAMA_API_KEY = "" 

CAP_OUTPUT_TPS = False
MAX_OUTPUT_TPS = 40

RESEARCH_FILE = "r.txt"
LOG_FILE = "r.text"

STREAM_MULTIPLIER = 0
SIDE_TELEMETRY = True

BASE_DIR = Path(__file__).resolve().parent
RESEARCH_PATH = BASE_DIR / RESEARCH_FILE
LOG_PATH = BASE_DIR / LOG_FILE


class A:
    R = "\x1b[0m"
    B = "\x1b[1m"
    D = "\x1b[2m"
    K = "\x1b[30m"
    R1 = "\x1b[31m"
    G = "\x1b[32m"
    Y = "\x1b[33m"
    B1 = "\x1b[34m"
    M = "\x1b[35m"
    C = "\x1b[36m"
    W = "\x1b[37m"
    GK = "\x1b[90m"
    RR = "\x1b[91m"
    GG = "\x1b[92m"
    YY = "\x1b[93m"
    BB = "\x1b[94m"
    MM = "\x1b[95m"
    CC = "\x1b[96m"
    WW = "\x1b[97m"


print_lock = threading.Lock()
STREAMING_ACTIVE = False  


def _enable_windows_vt():
    if os.name != "nt":
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if k.GetConsoleMode(h, ctypes.byref(mode)) == 0:
            return
        k.SetConsoleMode(h, mode.value | 0x0004)
    except Exception:
        pass


def _set_title(t):
    try:
        sys.stdout.write(f"\x1b]0;{t}\x07")
        sys.stdout.flush()
    except Exception:
        pass


def _term_size():
    try:
        ts = shutil.get_terminal_size((128, 34))
        return ts.columns, ts.lines
    except Exception:
        return 128, 34


def _clear():
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()


def _flush():
    try:
        sys.stdout.flush()
    except Exception:
        pass


def _sleep(s):
    time.sleep(max(0.0, s))


def _ts():
    now = datetime.datetime.now()
    return now.strftime("%H:%M:%S.") + f"{now.microsecond//1000:03d}"


def _sha256_bytes(b):
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _wrap(s, width):
    if width < 14:
        return [s[:width]]
    out = []
    line = ""
    for w in s.split(" "):
        if not line:
            line = w
        elif len(line) + 1 + len(w) <= width:
            line += " " + w
        else:
            out.append(line)
            line = w
    if line:
        out.append(line)
    return out


def _box(title, lines, width):
    w = max(60, min(width, 156))
    inner = w - 2
    top = "┏" + "━" * inner + "┓"
    bot = "┗" + "━" * inner + "┛"
    t = f" {title} "
    if len(t) > inner:
        t = t[:inner]
    mid = "┃" + t + " " * (inner - len(t)) + "┃"
    body = []
    for ln in lines:
        for seg in _wrap(ln, inner - 2):
            body.append("┃ " + seg + " " * (inner - 2 - len(seg)) + " ┃")
    if not body:
        body = ["┃" + " " * inner + "┃"]
    return [top, mid] + body + [bot]


def _safe_print(s, end="\n"):
    with print_lock:
        sys.stdout.write(s + end)
        _flush()


def _line(text="", color=A.GK, end="\n"):
    _safe_print(color + text + A.R, end=end)


def _hr(width, color=A.GK, ch="═"):
    _line(ch * max(28, width), color=color)


def _type_stream(text, color=A.GK, min_d=0.0006, max_d=0.006, end="\n"):
    with print_lock:
        for ch in text:
            sys.stdout.write(color + ch + A.R)
            _flush()
            _sleep(random.uniform(min_d, max_d))
        sys.stdout.write(end)
        _flush()


def _spinner(label, seconds=1.0, color=A.CC):
    frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    t0 = time.time()
    i = 0
    while time.time() - t0 < seconds:
        with print_lock:
            sys.stdout.write("\r" + color + frames[i % len(frames)] + A.R + " " + A.WW + label + A.R + " " * 12)
            _flush()
        _sleep(0.03)
        i += 1
    with print_lock:
        sys.stdout.write("\r" + " " * (len(label) + 32) + "\r")
        _flush()


def _prompt_prefix():
    user = getpass.getuser()
    host = platform.node() or socket.gethostname()
    cwd = str(BASE_DIR)
    if os.name == "nt":
        return f"{user}@{host} {cwd}> "
    return f"{user}@{host}:{cwd}$ "


class TokenBucket:
    def __init__(self, rate_tps):
        self.rate = float(rate_tps)
        self.t0 = time.time()
        self.tokens = 0.0

    def _estimate_tokens(self, s):
        if not s:
            return 0.0
        return max(1.0, len(s) / 4.0)

    def gate(self, s):
        if not CAP_OUTPUT_TPS:
            return
        need = self._estimate_tokens(s)
        self.tokens += need
        elapsed = time.time() - self.t0
        allowed = elapsed * self.rate
        if self.tokens > allowed:
            sleep_for = (self.tokens - allowed) / self.rate
            if sleep_for > 0:
                _sleep(min(0.25, sleep_for))


def _read_research():
    if not RESEARCH_PATH.exists():
        return "", {"exists": False, "bytes": 0, "lines": 0, "sha256": "", "truncated": False, "kept_bytes": 0}
    raw = b""
    try:
        raw = RESEARCH_PATH.read_bytes()
    except Exception:
        try:
            with open(RESEARCH_PATH, "rb") as f:
                raw = f.read()
        except Exception:
            return "", {"exists": True, "bytes": 0, "lines": 0, "sha256": "", "truncated": False, "kept_bytes": 0}

    total_bytes = len(raw)
    sha = _sha256_bytes(raw) if total_bytes else ""
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        text = str(raw)
    lines = text.count("\n") + (1 if text else 0)

    hard_limit_bytes = 180_000
    truncated = False
    kept = total_bytes
    if total_bytes > hard_limit_bytes:
        truncated = True
        kept = hard_limit_bytes
        raw = raw[-hard_limit_bytes:]
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = str(raw)

    return text, {"exists": True, "bytes": total_bytes, "lines": lines, "sha256": sha, "truncated": truncated, "kept_bytes": kept}


def _safe_append(path, text):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass


def _format_ns(ns):
    try:
        ns = int(ns)
    except Exception:
        return "?"
    if ns <= 0:
        return "0s"
    s = ns / 1e9
    if s < 1:
        return f"{s:.3f}s"
    if s < 60:
        return f"{s:.2f}s"
    m = int(s // 60)
    r = s - m * 60
    return f"{m}m {r:.1f}s"


def _system_style():
    return (
        "You are a careful scientific reasoning assistant. "
        "Be structured and grounded. Avoid fake citations. "
        "If uncertain, say so. Prioritize health impacts and safety, then efficiency, then practicality."
        " have a very long CoT to think out everything humanely possible, covering all angles, and then give a final verdict. Be honest and nuanced: the best choice depends on the scenario. "
    )


def _task_prompt():
    return (
        "Compare the most efficient shapes to store energy in batteries and liquid fuels.\n"
        "Decide what is most efficient AND healthiest overall. Be honest: it depends on the scenario.\n"
        "Cover: energy density (mass+volume), round-trip efficiency, conversion losses, infrastructure, storage stability, safety risks, and health impacts (air pollution vs mining/manufacturing).\n"
        "Give a clear verdict + a short 'best choice by scenario' list (cars, buses, generators, shipping, aviation, home backup).\n"
        "Use typical ranges if you include numbers and label assumptions.\n"
        "End with one sentence recommendation.\n"
        "Then finish with a final 4-8 word line choosing only: SHAPES (must) 'batteries' or 'liquid fuels'."
    )


def _is_gpt_oss(m):
    ml = str(m).lower()
    return ml.startswith("gpt-oss") or "gpt-oss" in ml


def _build_payload(research_text):
    msgs = [{"role": "system", "content": _system_style()}]
    if research_text.strip():
        msgs.append({"role": "user", "content": "Research notes (from local r.txt):\n" + research_text})
    msgs.append({"role": "user", "content": _task_prompt()})

    think_value = think_level if _is_gpt_oss(model) else True

    return {
        "model": model,
        "messages": msgs,
        "stream": True,
        # NOTE: Some servers/models may ignore this. It’s safe if unsupported.
        "think": think_value,
        "options": {
            "temperature": 0.2,
            "top_p": 0.92,
            "num_ctx": int(num_ctx),
            "num_predict": int(num_predict),
        }
    }


def _split_think_tags(text):
    out = []
    i = 0
    mode = "answer"
    while i < len(text):
        if text.startswith("<think>", i):
            if i > 0:
                out.append((mode, text[:i]))
            text = text[i+7:]
            i = 0
            mode = "thinking"
            continue
        if text.startswith("</think>", i):
            if i > 0:
                out.append((mode, text[:i]))
            text = text[i+8:]
            i = 0
            mode = "answer"
            continue
        i += 1
    if text:
        out.append((mode, text))
    return [(m, s) for (m, s) in out if s]


class OllamaClient:
    def __init__(self, use_cloud: bool, local_base: str, cloud_base: str, api_key: str):
        self.use_cloud = bool(use_cloud)
        self.base = (cloud_base if self.use_cloud else local_base).rstrip("/")
        self.api = self.base + "/api"
        self.requires_key = self.use_cloud

        key = (api_key or "").strip()
        if not key:
            key = (os.environ.get("OLLAMA_API_KEY") or "").strip()
        self.key = key

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.use_cloud and self.key:
            h["Authorization"] = "Bearer " + self.key
        return h

    def tags(self, timeout=10.0):
        url = self.api + "/tags"
        req = urllib.request.Request(url, method="GET", headers=self._headers())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def chat_stream(self, payload, timeout=300.0):
        url = self.api + "/chat"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST", headers=self._headers())
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line.decode("utf-8"))
                except Exception:
                    continue


def _trace(ch, msg, color=A.GK):
    _line(f"[{_ts()}] [{ch:<10}] {msg}", color=color)


def _hex_dump(lines=10, width=110, color=A.GK):
    base_addr = random.randint(0x1000, 0xFFFFF)
    for i in range(lines):
        addr = base_addr + i * 16
        bs = [random.randint(0, 255) for _ in range(16)]
        hx = " ".join(f"{b:02X}" for b in bs)
        asc = "".join(chr(b) if 32 <= b <= 126 else "." for b in bs)
        s = f"{addr:08X}  {hx:<47}  |{asc}|"
        _line(s[:width], color=color)
        _sleep(random.uniform(0.008, 0.02))


def _burst_streams(meta, client: OllamaClient):
    cols, _ = _term_size()
    host = client.base
    osx = platform.system() + " " + platform.release()
    arch = platform.machine() or "unknown"
    pid = os.getpid()
    key_state = "present" if (client.key.strip() if client.key else "") else "missing"
    mode = "cloud" if client.use_cloud else "local"

    channels = [
        ("SYS", A.GK), ("BUS", A.CC), ("FS", A.MM), ("NET", A.CC), ("TLS", A.BB),
        ("AUTH", A.YY), ("MUX", A.BB), ("PIPE", A.BB), ("CACHE", A.MM), ("MEM", A.MM),
        ("LAT", A.CC), ("QOS", A.YY), ("DIAG", A.GG), ("IDX", A.MM), ("PROMPT", A.BB),
        ("MODEL", A.YY), ("THINK", A.GK), ("ANSWER", A.WW), ("LOG", A.GG), ("STATS", A.YY),
        ("KERNEL", A.GK), ("SCHED", A.GK), ("IO", A.CC), ("TIME", A.GK),
    ]

    sys_msgs = [
        f"os={osx} arch={arch} pid={pid}",
        f"tty={cols}x? vt=on unicode=on",
        "io=nonblocking mux=enabled channels=24",
        "policy=local-files-only audit=enabled",
    ]
    fs_msgs = [
        f"cwd={BASE_DIR}",
        f"research={RESEARCH_FILE} exists={meta.get('exists')} bytes={meta.get('bytes')} lines={meta.get('lines')}",
        f"research_sha256={meta.get('sha256') or 'n/a'}",
        f"research_truncated={meta.get('truncated')} loaded_bytes={meta.get('kept_bytes')}",
        f"log={LOG_FILE} mode=append",
    ]
    net_msgs = [
        f"remote={host} api=/api/chat mode={mode}",
        "transport=http(s) keepalive=on",
        "stream=chunked framing=jsonl",
        "timeouts=nominal retry=off",
    ]
    tls_msgs = [
        "tls=enabled (if https)",
        "cert=system trust=ok",
        "ciphersuite=default",
        "handshake=on-demand",
    ]
    auth_msgs = [
        f"api_key={key_state} (cloud only)",
        "auth=Bearer header (cloud only)",
        "scope=cloud-api",
        "revocation=server-side",
    ]
    model_msgs = [
        f"model={model}",
        f"think={'high' if _is_gpt_oss(model) else 'true'}",
        f"num_ctx={num_ctx} num_predict={num_predict}",
        "temperature=0.2 top_p=0.92",
    ]
    diag_msgs = [
        "healthcheck=armed",
        "integrity=ok tamper=none",
        "trace_clock=ms",
        "stdout_flush=aggressive",
    ]

    msg_map = {
        "SYS": sys_msgs,
        "FS": fs_msgs,
        "NET": net_msgs,
        "TLS": tls_msgs,
        "AUTH": auth_msgs,
        "MODEL": model_msgs,
        "DIAG": diag_msgs,
    }

    _trace("SYS", "console bootstrap", A.CC)
    _hex_dump(lines=min(6, 3 + STREAM_MULTIPLIER), width=min(cols, 120), color=A.GK)

    total_ticks = 45 + (STREAM_MULTIPLIER * 20)
    for _ in range(total_ticks):
        ch, col = random.choice(channels)
        bank = msg_map.get(ch, None)
        if bank:
            msg = random.choice(bank)
        else:
            msg = random.choice([
                "buffer=ok",
                "queue depth nominal",
                "mux slot rotated",
                "frame decode ok",
                "pressure=stable",
                "tick",
                "event=accepted",
                "sync",
                "heartbeat",
            ])
        _trace(ch, msg, col)
        _sleep(random.uniform(0.002, 0.03))
    _line("")


def _render_header(meta, client: OllamaClient):
    w, _ = _term_size()
    _clear()
    mode = "CLOUD" if client.use_cloud else "LOCAL"
    _set_title(f"Energy Storage // {mode} Reasoning Console")
    title = f"ENERGY STORAGE COMPARISON CONSOLE // {mode}"
    sub = "liquid fuels vs batteries | research ingest | telemetry mux"
    lines = [
        f"id: {code_id}",
        f"mode: {'cloud' if client.use_cloud else 'local'}",
        f"endpoint: {client.base}/api",
        f"model: {model}",
        f"think: {_is_gpt_oss(model)} level={think_level}",
        f"context: num_ctx={num_ctx}  generation: num_predict={num_predict}",
        f"output cap: {('on ' + str(MAX_OUTPUT_TPS) + ' TPS') if CAP_OUTPUT_TPS else 'off'}",
        f"research input: {RESEARCH_FILE} ({'present' if meta.get('exists') else 'missing'})",
        f"log output: {LOG_FILE}",
        f"time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    _line(title.center(min(w, 130)), A.CC)
    _line(sub.center(min(w, 130)), A.GK)
    _hr(min(w, 130), A.GK)
    for ln in _box("STATUS", lines, min(w, 130)):
        _line(ln, A.BB)
    _line("")
    _line("Commands: /help  /start  /models  /setmodel <name>  /cap on|off  /cap <tps>  /research  /openlog  /clear  /quit", A.GK)
    _line("")


def _help():
    cmds = [
        "/start                 run analysis (thinking grey + answer white)",
        "/models                list models on the server",
        "/setmodel <name>       set model for this session",
        "/cap on|off            toggle output TPS cap",
        "/cap <tps>             set output TPS cap (display throttle)",
        "/research              show r.txt stats",
        "/openlog               show tail of r.text",
        "/clear                 redraw header",
        "/quit                  exit"
    ]
    w, _ = _term_size()
    for ln in _box("COMMANDS", cmds, min(w, 130)):
        _line(ln, A.CC)
    _line("")


def _print_research_stats(meta):
    w, _ = _term_size()
    lines = []
    if not meta.get("exists"):
        lines.append("r.txt is missing. Create r.txt and put your research notes in it.")
    else:
        lines.append(f"bytes={meta['bytes']} lines={meta['lines']}")
        lines.append(f"sha256={meta['sha256']}")
        lines.append(f"loaded_bytes={meta['kept_bytes']} truncated={'yes' if meta['truncated'] else 'no'}")
    for ln in _box("RESEARCH INPUT (r.txt)", lines, min(w, 130)):
        _line(ln, A.MM)
    _line("")


def _tail_log(n=80):
    try:
        if not LOG_PATH.exists():
            _line("r.text not found yet. Run /start first.", A.RR)
            _line("")
            return
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            data = f.read().splitlines()
        chunk = data[-n:]
        w, _ = _term_size()
        for ln in _box("r.text (tail)", chunk, min(w, 130)):
            _line(ln, A.GK)
        _line("")
    except Exception as e:
        _line(f"Failed to read r.text: {e}", A.RR)
        _line("")


def _list_models(client: OllamaClient):
    try:
        _spinner("querying /api/tags", 0.22, A.CC)
        data = client.tags(timeout=15.0)
        models = data.get("models", [])
        names = []
        for m in models:
            n = m.get("name", "")
            if n:
                names.append(n)
        names = sorted(set(names), key=lambda x: x.lower())
        if not names:
            _line("No models returned.", A.RR)
            _line("")
            return
        w, _ = _term_size()
        for ln in _box("MODELS", names[:220], min(w, 130)):
            _line(ln, A.GG)
        _line("")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            _line("Auth failed listing models. If you are using cloud, check your API key.", A.RR)
        else:
            _line(f"HTTP error listing models: {e}", A.RR)
        _line("")
    except Exception as e:
        _line(f"Failed to fetch models: {e}", A.RR)
        _line("")


def _write_log(meta, prompt_dump, thinking_text, answer_text, final_stats, client: OllamaClient):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    td = final_stats.get("total_duration")
    ed = final_stats.get("eval_duration")
    pd = final_stats.get("prompt_eval_duration")
    ec = final_stats.get("eval_count")
    pc = final_stats.get("prompt_eval_count")

    header = "\n" + ("=" * 110) + "\n"
    block = header
    block += f"time: {stamp}\n"
    block += f"mode: {'cloud' if client.use_cloud else 'local'}\n"
    block += f"endpoint: {client.base}/api\n"
    block += f"model: {model}\n"
    block += f"think: {think_level if _is_gpt_oss(model) else 'true'}\n"
    block += f"num_ctx: {num_ctx} | num_predict: {num_predict}\n"
    block += f"research: {RESEARCH_FILE} exists={meta.get('exists')} bytes={meta.get('bytes')} lines={meta.get('lines')} truncated={meta.get('truncated')} loaded_bytes={meta.get('kept_bytes')}\n"
    block += f"research_sha256: {meta.get('sha256')}\n"
    block += f"durations: total={_format_ns(td)} prompt_eval={_format_ns(pd)} eval={_format_ns(ed)}\n"
    block += f"tokens: prompt={pc} eval={ec}\n\n"
    block += "PROMPT:\n" + prompt_dump.strip() + "\n\n"
    block += "THINKING TRACE:\n" + (thinking_text.strip() if thinking_text.strip() else "[none]") + "\n\n"
    block += "FINAL ANSWER:\n" + (answer_text.strip() if answer_text.strip() else "[none]") + "\n"
    _safe_append(LOG_PATH, block)


def _telemetry_thread(stop_evt):
    global STREAMING_ACTIVE
    channels = [
        ("BUS", A.CC), ("NET", A.CC), ("TLS", A.BB), ("PIPE", A.BB), ("CACHE", A.MM),
        ("LAT", A.CC), ("QOS", A.YY), ("SCHED", A.GK), ("IO", A.CC), ("DIAG", A.GG),
        ("WATCH", A.YY), ("MUX", A.BB), ("STATS", A.YY), ("LOG", A.GG),
    ]
    msgs = [
        "frame ok", "queue stable", "backpressure nominal", "mux rotate", "heartbeat",
        "decode ok", "flush", "keepalive", "latency window", "qos ok", "buffer ok",
        "trace tick", "event", "sync", "commit", "checkpoint",
    ]
    while not stop_evt.is_set():
        if STREAMING_ACTIVE:
            _sleep(0.05)
            continue
        ch, col = random.choice(channels)
        msg = random.choice(msgs)
        got = print_lock.acquire(timeout=0.02)
        if got:
            try:
                sys.stdout.write(col + f"[{_ts()}] [{ch:<10}] " + A.R + A.GK + msg + A.R + "\n")
                _flush()
            finally:
                print_lock.release()
        _sleep(random.uniform(0.01, 0.06))


def _run_start(client: OllamaClient):
    global STREAMING_ACTIVE
    research_text, meta = _read_research()
    STREAMING_ACTIVE = False
    _line("run:", A.GK)
    _spinner("arming telemetry mux", 0.22, A.CC)
    _burst_streams(meta, client)

    if client.requires_key and not client.key:
        _line("Cloud API key missing. Set env OLLAMA_API_KEY (or toggle USE_CLOUD=False).", A.RR)
        _line("")
        return

    prompt_dump = _system_style() + "\n\n"
    if research_text.strip():
        prompt_dump += f"[Research from {RESEARCH_FILE}]\n" + research_text + "\n\n"
    prompt_dump += "[Task]\n" + _task_prompt()

    payload = _build_payload(research_text)

    thinking_buf = []
    answer_buf = []
    final_stats = {}

    bucket = TokenBucket(MAX_OUTPUT_TPS)

    printed_thinking = False
    printed_answer = False

    def thinking_header():
        nonlocal printed_thinking
        if printed_thinking:
            return
        printed_thinking = True
        _line("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓", A.GK)
        _line("┃ THINKING TRACE (stream)                                                                           ┃", A.GK)
        _line("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛", A.GK)

    def answer_header():
        nonlocal printed_answer
        if printed_answer:
            return
        printed_answer = True
        _line("")
        _line("ANSWER:", A.WW)

    stop_evt = threading.Event()
    if SIDE_TELEMETRY:
        threading.Thread(target=_telemetry_thread, args=(stop_evt,), daemon=True).start()

    _line("")
    _line("stream: thinking=grey | answer=white", A.GK)
    _line("")

    STREAMING_ACTIVE = True  # pause fake telemetry while the model is printing
    try:
        for chunk in client.chat_stream(payload, timeout=600.0):
            if "error" in chunk:
                stop_evt.set()
                _line(f"ollama error: {chunk.get('error')}", A.RR)
                break

            msg = chunk.get("message") or {}
            ttxt = msg.get("thinking") or ""
            ctxt = msg.get("content") or ""

            if ttxt:
                if not printed_thinking:
                    thinking_header()
                bucket.gate(ttxt)
                with print_lock:
                    sys.stdout.write(A.GK + ttxt + A.R)
                    _flush()
                thinking_buf.append(ttxt)

            if ctxt:
                parts = _split_think_tags(ctxt)
                for mode_name, seg in parts:
                    if mode_name == "thinking":
                        if not printed_thinking:
                            thinking_header()
                        bucket.gate(seg)
                        with print_lock:
                            sys.stdout.write(A.GK + seg + A.R)
                            _flush()
                        thinking_buf.append(seg)
                    else:
                        if not printed_answer:
                            answer_header()
                        bucket.gate(seg)
                        with print_lock:
                            sys.stdout.write(A.WW + seg + A.R)
                            _flush()
                        answer_buf.append(seg)

            if chunk.get("done"):
                final_stats = chunk
                stop_evt.set()
                break

        stop_evt.set()
        STREAMING_ACTIVE = False
        _line("\n")

        td = final_stats.get("total_duration")
        ed = final_stats.get("eval_duration")
        pd = final_stats.get("prompt_eval_duration")
        ec = final_stats.get("eval_count")
        pc = final_stats.get("prompt_eval_count")

        w, _ = _term_size()
        metrics = [
            f"model={model}",
            f"think={think_level if _is_gpt_oss(model) else 'true'}",
            f"num_ctx={num_ctx} num_predict={num_predict}",
            f"research present={meta.get('exists')} truncated={meta.get('truncated')}",
            f"total={_format_ns(td)} prompt_eval={_format_ns(pd)} eval={_format_ns(ed)}",
            f"tokens: prompt={pc} eval={ec}",
            f"log_write={LOG_FILE}"
        ]
        for ln in _box("RUN METRICS", metrics, min(w, 130)):
            _line(ln, A.YY)

        _line("")
        _trace("LOG", "writing research log (r.text)", A.GG)
        _write_log(meta, prompt_dump, "".join(thinking_buf), "".join(answer_buf), final_stats, client)
        _trace("LOG", "saved → r.text", A.GG)
        _line("")

    except urllib.error.HTTPError as e:
        stop_evt.set()
        STREAMING_ACTIVE = False
        if e.code in (401, 403):
            _line("Auth failed. If using cloud, your API key is wrong/missing/revoked.", A.RR)
        else:
            _line(f"HTTP error: {e}", A.RR)
        _line("")
    except urllib.error.URLError as e:
        stop_evt.set()
        STREAMING_ACTIVE = False
        _line(f"Network error: {e}", A.RR)
        _line("Tip: is Ollama running? Local default is http://127.0.0.1:11434", A.GK)
        _line("")
    except KeyboardInterrupt:
        stop_evt.set()
        STREAMING_ACTIVE = False
        _line("\nInterrupted.", A.RR)
        _line("")
    except Exception as e:
        stop_evt.set()
        STREAMING_ACTIVE = False
        _line(f"Crash: {e}", A.RR)
        _line("")


def _parse_cmd(s):
    s = s.strip()
    if not s:
        return ("", [])
    parts = s.split()
    return (parts[0].lower(), parts[1:])


def main():
    global model, CAP_OUTPUT_TPS, MAX_OUTPUT_TPS, think_level
    _enable_windows_vt()
    random.seed()

    api_key = (OLLAMA_API_KEY or os.environ.get("OLLAMA_API_KEY") or "").strip()
    client = OllamaClient(USE_CLOUD, LOCAL_BASE, CLOUD_BASE, api_key)

    _, meta = _read_research()
    _render_header(meta, client)

    while True:
        try:
            with print_lock:
                sys.stdout.write(A.GG + _prompt_prefix() + A.R)
                _flush()
            raw = sys.stdin.readline()
            if not raw:
                break
            cmd, args = _parse_cmd(raw)

            if cmd in ("/quit", "/exit"):
                _type_stream("session closed.", A.GK, 0.0007, 0.004)
                break

            if cmd == "/help":
                _help()
                continue

            if cmd == "/clear":
                _, meta = _read_research()
                _render_header(meta, client)
                continue

            if cmd == "/models":
                _list_models(client)
                continue

            if cmd == "/setmodel":
                if not args:
                    _line("Usage: /setmodel <name>", A.RR)
                    _line("")
                    continue
                model = " ".join(args).strip()
                _line(f"model set: {model}", A.YY)
                _line("")
                continue

            if cmd == "/cap":
                if not args:
                    _line("Usage: /cap on|off OR /cap <tps>", A.RR)
                    _line("")
                    continue
                v = args[0].lower()
                if v in ("on", "true", "1"):
                    CAP_OUTPUT_TPS = True
                    _line(f"output cap: ON ({MAX_OUTPUT_TPS} TPS)", A.YY)
                elif v in ("off", "false", "0"):
                    CAP_OUTPUT_TPS = False
                    _line("output cap: OFF", A.YY)
                else:
                    try:
                        MAX_OUTPUT_TPS = max(1, int(v))
                        CAP_OUTPUT_TPS = True
                        _line(f"output cap: ON ({MAX_OUTPUT_TPS} TPS)", A.YY)
                    except Exception:
                        _line("Invalid. Use: /cap on|off or /cap 60", A.RR)
                _line("")
                continue

            if cmd == "/research":
                _, meta = _read_research()
                _print_research_stats(meta)
                continue

            if cmd == "/openlog":
                _tail_log(80)
                continue

            if cmd == "/start":
                if _is_gpt_oss(model):
                    think_level = "high"
                _run_start(client)
                continue

            if cmd == "":
                continue

            if cmd.startswith("/"):
                _line(f"Unknown command: {cmd}. Try /help", A.RR)
                _line("")
                continue

            _line("This console only reacts to slash-commands. Try /start.", A.GK)
            _line("")

        except KeyboardInterrupt:
            _line("\nCtrl+C caught. Type /quit to exit.", A.RR)
            _line("")
        except Exception as e:
            _line(f"Loop error: {e}", A.RR)
            _line("")


if __name__ == "__main__":
    main()
