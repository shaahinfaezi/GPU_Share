#!/usr/bin/env python3
"""Turn a saved results directory into tables and figures -- offline.

    python render_results.py results/T1-20260826-141500
    python render_results.py results/T1-*  --out artifacts/

Requirement 3 in one sentence: a chart you cannot redraw is not evidence.
Prometheus retention will expire, the Grafana dashboard will be edited, and the
cluster will be rebuilt. Everything needed to redraw every figure was written
into the results directory at run time, and this script reads ONLY that -- no
cluster, no network, no Prometheus. If it runs a year from now on a laptop, the
numbers are reproducible.

Two independent sources are reported side by side on purpose:

  CLIENT   replay.requests.csv / replay.summary.json. Per-request wall-clock
           TTFT, inter-token latency and end-to-end time, measured outside the
           cluster. This is authoritative for latency: it is what a caller
           actually experienced, including queueing the server never sees.

  SERVER   prom.json, from vLLM's own histograms plus HAMi and DCGM. This is
           what the dashboard shows and therefore what the video shows.

Where they disagree, the gap is itself a finding -- it is queue time upstream
of the engine. Round one could not see that gap at all, because it was not
streaming and had only one clock.

Outputs, all self-contained:
    summary.csv     one row per (sub-run, role); the numbers behind every claim
    arms.csv        the arm-vs-arm comparison, with deltas against dedicated
    report.html     tables plus inline SVG figures, no external assets
"""
import argparse
import csv
import glob
import html
import json
import math
import os
import sys

# --------------------------------------------------------------------- theme
# Chosen to survive both a light projector and a dark screen recording.
INK = "#1a1d23"
MUTE = "#6b7280"
GRID = "#e4e7ec"
ARM_COLOR = {"dedicated": "#5b6b7f", "hami": "#2f7ed8", "hami-mps": "#e8710a"}
SERIES = ["#2f7ed8", "#e8710a", "#1aa260", "#a33bc2", "#c0392b", "#0aa5b5"]


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


# ------------------------------------------------------------------ svg bits
def _esc(s):
    return html.escape(str(s), quote=True)


def _fmt(v, unit=""):
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if unit == "ms":
        return "%.0f" % (v * 1000.0)
    if unit == "s":
        return "%.2f" % v
    if abs(v) >= 100:
        return "%.0f" % v
    if abs(v) >= 10:
        return "%.1f" % v
    return "%.2f" % v


def grouped_bars(title, groups, series, values, unit="", note=""):
    """groups = x categories; series = legend entries; values[s][g] = number."""
    W, H = 860, 330
    L, R, T, B = 66, 16, 44, 64
    pw, ph = W - L - R, H - T - B
    flat = [v for s in series for v in values.get(s, {}).values() if v is not None]
    if not flat:
        return ""
    vmax = max(flat) * 1.18 or 1.0
    gw = pw / max(1, len(groups))
    bw = gw / (len(series) + 0.6)

    p = ['<svg viewBox="0 0 %d %d" class="fig" role="img" aria-label="%s">'
         % (W, H, _esc(title))]
    p.append('<text x="0" y="18" class="ttl">%s</text>' % _esc(title))
    if note:
        p.append('<text x="0" y="34" class="sub">%s</text>' % _esc(note))
    for i in range(5):
        y = T + ph - ph * i / 4.0
        p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="grid"/>'
                 % (L, y, W - R, y))
        p.append('<text x="%d" y="%.1f" class="ax" text-anchor="end">%s</text>'
                 % (L - 8, y + 4, _fmt(vmax * i / 4.0, unit)))
    for gi, g in enumerate(groups):
        for si, s in enumerate(series):
            v = values.get(s, {}).get(g)
            if v is None:
                continue
            h = max(1.0, ph * (v / vmax))
            x = L + gi * gw + 0.3 * bw + si * bw
            y = T + ph - h
            col = ARM_COLOR.get(s, SERIES[si % len(SERIES)])
            p.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                     'fill="%s" rx="2"><title>%s / %s: %s</title></rect>'
                     % (x, y, bw * 0.86, h, col, _esc(s), _esc(g),
                        _fmt(v, unit)))
            if h > 26:
                p.append('<text x="%.1f" y="%.1f" class="val" '
                         'text-anchor="middle">%s</text>'
                         % (x + bw * 0.43, y - 4, _fmt(v, unit)))
        p.append('<text x="%.1f" y="%d" class="ax" text-anchor="middle">%s</text>'
                 % (L + gi * gw + gw / 2, T + ph + 18, _esc(g)))
    for si, s in enumerate(series):
        col = ARM_COLOR.get(s, SERIES[si % len(SERIES)])
        x = L + si * 132
        p.append('<rect x="%d" y="%d" width="11" height="11" fill="%s" rx="2"/>'
                 % (x, H - 26, col))
        p.append('<text x="%d" y="%d" class="ax">%s</text>'
                 % (x + 16, H - 17, _esc(s)))
    p.append('</svg>')
    return "".join(p)


def lines(title, series, unit="", note="", ylab=""):
    """series = [(label, [(t, v), ...]), ...] with t already relative seconds."""
    series = [(lab, pts) for lab, pts in series if pts]
    if not series:
        return ""
    W, H = 860, 300
    L, R, T, B = 66, 120, 44, 40
    pw, ph = W - L - R, H - T - B
    xs = [t for _, pts in series for t, _ in pts]
    ys = [v for _, pts in series for _, v in pts]
    x0, x1 = min(xs), max(xs)
    y1 = max(ys) * 1.12 or 1.0
    y0 = 0.0
    if x1 <= x0:
        x1 = x0 + 1

    def px(t):
        return L + pw * (t - x0) / (x1 - x0)

    def py(v):
        return T + ph - ph * (v - y0) / (y1 - y0 or 1)

    p = ['<svg viewBox="0 0 %d %d" class="fig" role="img" aria-label="%s">'
         % (W, H, _esc(title))]
    p.append('<text x="0" y="18" class="ttl">%s</text>' % _esc(title))
    if note:
        p.append('<text x="0" y="34" class="sub">%s</text>' % _esc(note))
    for i in range(5):
        y = T + ph - ph * i / 4.0
        p.append('<line x1="%d" y1="%.1f" x2="%.1f" y2="%.1f" class="grid"/>'
                 % (L, y, L + pw, y))
        p.append('<text x="%d" y="%.1f" class="ax" text-anchor="end">%s</text>'
                 % (L - 8, y + 4, _fmt(y1 * i / 4.0, unit)))
    for i in range(6):
        t = x0 + (x1 - x0) * i / 5.0
        p.append('<text x="%.1f" y="%d" class="ax" text-anchor="middle">%ds</text>'
                 % (px(t), T + ph + 18, int(t)))
    for si, (lab, pts) in enumerate(series):
        col = SERIES[si % len(SERIES)]
        d = " ".join("%s%.1f %.1f" % ("M" if i == 0 else "L", px(t), py(v))
                     for i, (t, v) in enumerate(pts))
        p.append('<path d="%s" fill="none" stroke="%s" stroke-width="1.8" '
                 'stroke-linejoin="round"/>' % (d, col))
        p.append('<rect x="%d" y="%d" width="10" height="10" fill="%s" rx="2"/>'
                 % (L + pw + 12, T + 6 + si * 20, col))
        p.append('<text x="%d" y="%d" class="ax">%s</text>'
                 % (L + pw + 27, T + 15 + si * 20, _esc(lab[:16])))
    if ylab:
        p.append('<text x="0" y="%d" class="sub">%s</text>' % (T - 2, _esc(ylab)))
    p.append('</svg>')
    return "".join(p)


# ------------------------------------------------------------- data plumbing
def prom_series(prom, key, label_key="r2_role", t_origin=None):
    """Flatten one saved Prometheus query into [(label, [(t, v)...]), ...]."""
    node = (prom or {}).get("series", {}).get(key) or {}
    out = []
    for res in node.get("result", []) or []:
        m = res.get("metric", {})
        lab = m.get(label_key) or m.get("gpu") or m.get("device_index") \
            or m.get("instance") or key
        pts = []
        for t, v in res.get("values", []) or []:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(fv):
                continue
            pts.append((float(t) - (t_origin or 0), fv))
        if pts:
            out.append((str(lab), pts))
    return out


def gpu_csv(path, t_origin=None):
    """1 Hz nvidia-smi -> {device: [(t, mib), ...]}, plus util."""
    mem, util = {}, {}
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    t = float(row["t"]) - (t_origin or 0)
                    idx = "GPU " + str(row["index"]).strip()
                    mem.setdefault(idx, []).append((t, float(row["memory.used"])))
                    util.setdefault(idx, []).append(
                        (t, float(row["utilization.gpu"])))
                except Exception:
                    continue
    except Exception:
        pass
    return mem, util


def collect(run_dirs):
    runs = []
    for rd in run_dirs:
        doc = read_json(os.path.join(rd, "run.json"))
        if not doc:
            print("  skip %s (no run.json)" % rd, file=sys.stderr)
            continue
        doc["_dir"] = rd
        subs = []
        for sub in doc.get("subruns", []):
            tag = sub.get("tag") or ("%s__scaledown" % sub.get("arm", "?"))
            sd = os.path.join(rd, tag)
            sub["_dir"] = sd
            sub.setdefault("summary", read_json(
                os.path.join(sd, "replay.summary.json"), {}) or {})
            sub["_prom"] = read_json(os.path.join(sd, "prom.json"), {}) or {}
            sub["_gpu"] = os.path.join(sd, "gpu.csv")
            subs.append(sub)
        doc["subruns"] = subs
        runs.append(doc)
    return runs


ROWS = [
    ("requests", lambda b: b.get("requests")),
    ("failed", lambda b: b.get("failed")),
    ("tokens_per_second", lambda b: b.get("tokens_per_second")),
    ("requests_per_second", lambda b: b.get("requests_per_second")),
    ("ttft_mean_ms", lambda b: _g(b, "ttft", "mean", 1000)),
    ("ttft_p50_ms", lambda b: _g(b, "ttft", "p50", 1000)),
    ("ttft_p95_ms", lambda b: _g(b, "ttft", "p95", 1000)),
    ("ttft_p99_ms", lambda b: _g(b, "ttft", "p99", 1000)),
    ("itl_mean_ms", lambda b: _g(b, "itl", "mean", 1000)),
    ("itl_p95_ms", lambda b: _g(b, "itl", "p95", 1000)),
    ("itl_p99_ms", lambda b: _g(b, "itl", "p99", 1000)),
    ("ttlt_mean_s", lambda b: _g(b, "ttlt", "mean", 1)),
    ("ttlt_p95_s", lambda b: _g(b, "ttlt", "p95", 1)),
    ("ttlt_p99_s", lambda b: _g(b, "ttlt", "p99", 1)),
    ("e2e_p99_s", lambda b: _g(b, "latency_e2e", "p99", 1)),
    ("client_delay_p99_ms", lambda b: _g(b, "client_delay", "p99", 1000)),
]


def _g(block, name, stat, mult=1):
    sub = (block or {}).get(name) or {}
    v = sub.get(stat)
    return None if v is None else v * mult


def write_summary_csv(runs, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["test", "arm", "tag", "role", "kind", "trace_id", "status"]
                   + [n for n, _ in ROWS])
        n = 0
        for run in runs:
            for sub in run["subruns"]:
                s = sub.get("summary") or {}
                per = s.get("per_service") or {}
                blocks = list(per.items())
                if s.get("aggregate"):
                    blocks.append(("ALL", s["aggregate"]))
                for role, b in blocks:
                    w.writerow([run.get("test"), sub.get("arm"), sub.get("tag"),
                                role, (b or {}).get("kind", ""),
                                sub.get("trace_id_seen") or sub.get("trace_id"),
                                sub.get("status")]
                               + [_fmt(f(b or {})) for _, f in ROWS])
                    n += 1
    return n


def write_arms_csv(runs, path):
    """The comparison that answers the question, with dedicated as the base."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["test", "role", "metric", "dedicated", "hami", "hami-mps",
                    "hami_vs_dedicated_%", "mps_vs_hami_%"])
        for run in runs:
            by = {}
            for sub in run["subruns"]:
                per = (sub.get("summary") or {}).get("per_service") or {}
                for role, b in per.items():
                    by.setdefault(role, {})[sub.get("arm")] = b
            for role in sorted(by):
                for name, f in ROWS:
                    d = _num(f(by[role].get("dedicated") or {}))
                    h = _num(f(by[role].get("hami") or {}))
                    m = _num(f(by[role].get("hami-mps") or {}))
                    if d is None and h is None and m is None:
                        continue
                    w.writerow([run.get("test"), role, name,
                                _fmt(d), _fmt(h), _fmt(m),
                                _pct(h, d), _pct(m, h)])


def _num(v):
    return v if isinstance(v, (int, float)) else None


def _pct(a, b):
    if a is None or b in (None, 0):
        return ""
    return "%+.1f" % ((a - b) / b * 100.0)


# ------------------------------------------------------------------- report
CSS = """
:root{--ink:#1a1d23;--mute:#6b7280;--grid:#e4e7ec;--bg:#ffffff;--card:#f7f8fa;
--line:#dfe3e8;--warn:#b45309;--bad:#b91c1c;--good:#15803d}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
--ink:#e6e8ec;--mute:#9aa3af;--grid:#2c313a;--bg:#14171c;--card:#1b1f26;
--line:#2c313a;--warn:#f59e0b;--bad:#f87171;--good:#4ade80}}
:root[data-theme=dark]{--ink:#e6e8ec;--mute:#9aa3af;--grid:#2c313a;--bg:#14171c;
--card:#1b1f26;--line:#2c313a;--warn:#f59e0b;--bad:#f87171;--good:#4ade80}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;padding:32px 26px 80px;
font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
max-width:960px;margin-inline:auto}
h1{font-size:25px;margin:0 0 4px;letter-spacing:-.015em}
h2{font-size:18px;margin:38px 0 10px;padding-top:14px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:22px 0 8px;color:var(--mute);font-weight:600;
text-transform:uppercase;letter-spacing:.06em}
p{margin:9px 0}
.lede{color:var(--mute);margin-bottom:20px}
.meta{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:12px 14px;font:12.5px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;
color:var(--mute);overflow-x:auto}
.wrap{overflow-x:auto;margin:12px 0}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 9px;text-align:right;border-bottom:1px solid var(--line);
white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--mute);font-weight:600;font-size:11.5px;
text-transform:uppercase;letter-spacing:.05em}
tbody tr:hover{background:var(--card)}
.fig{width:100%;height:auto;display:block;margin:14px 0 4px}
.ttl{font:600 14px ui-sans-serif,sans-serif;fill:var(--ink)}
.sub{font:11.5px ui-sans-serif,sans-serif;fill:var(--mute)}
.ax{font:11px ui-sans-serif,sans-serif;fill:var(--mute)}
.val{font:10.5px ui-sans-serif,sans-serif;fill:var(--mute)}
.grid{stroke:var(--grid);stroke-width:1}
.note{border-left:3px solid var(--warn);padding:8px 0 8px 13px;color:var(--mute);
margin:14px 0;font-size:13.5px}
.bad{color:var(--bad);font-weight:600}
.good{color:var(--good)}
code{font:12.5px ui-monospace,Menlo,monospace;background:var(--card);
padding:1px 5px;border-radius:4px}
"""


def table(headers, rows):
    h = "".join("<th>%s</th>" % _esc(x) for x in headers)
    body = []
    for r in rows:
        body.append("<tr>" + "".join("<td>%s</td>" % _esc(c) for c in r) + "</tr>")
    return ('<div class="wrap"><table><thead><tr>%s</tr></thead><tbody>%s'
            '</tbody></table></div>' % (h, "".join(body)))


def build_report(runs, out_path):
    P = ['<title>Round 2 Results</title><style>%s</style>' % CSS]
    P.append("<h1>GPU sharing, round 2 — measured results</h1>")
    P.append('<p class="lede">Rendered offline from saved run data. No cluster '
             'or Prometheus was contacted; every number below came out of the '
             'results directory, so this page redraws identically a year from '
             'now.</p>')

    for run in runs:
        test = run.get("test", "?")
        cfg = run.get("config") or {}
        P.append("<h2>%s — %s</h2>" % (_esc(test),
                                       _esc(cfg.get("scenario", "scale-down"))))
        if cfg.get("why"):
            P.append("<p>%s</p>" % _esc(cfg["why"]))
        P.append('<div class="meta">trace_id %s &nbsp;·&nbsp; rate %s/s '
                 '&nbsp;·&nbsp; %ss &nbsp;·&nbsp; source %s &nbsp;·&nbsp; '
                 'seed %s<br>pinned to %s &nbsp;·&nbsp; pooling via %s'
                 '</div>'
                 % (_esc(run.get("trace_id")), _esc(cfg.get("rate", "—")),
                    _esc(cfg.get("duration", "—")), _esc(cfg.get("source", "—")),
                    _esc(run.get("seed", "—")),
                    _esc((run.get("pin_gpu") or "not pinned")[:24]),
                    _esc(run.get("pooling_via", "—"))))

        if run.get("trace_id_mismatch"):
            P.append('<p class="note bad">FAIRNESS VIOLATION — these arms did '
                     'not replay the same trace: %s. The comparison below is '
                     'not valid.</p>' % _esc(json.dumps(run["trace_id_mismatch"])))

        # T9 has a timeline rather than a throughput comparison
        for sub in run["subruns"]:
            if sub.get("timeline"):
                P.append("<h3>Admission timeline</h3>")
                P.append(table(["phase", "duration"],
                               list(sub["timeline"].items())))
                if sub.get("admitted_without_eviction"):
                    P.append('<p class="note">The newcomer was admitted '
                             'WITHOUT eviction, so the card was not actually '
                             'full. Re-check the gpumem budget before quoting '
                             'this run.</p>')
                reasons = sub.get("arrival_reasons") or []
                if reasons:
                    last = reasons[-1]
                    P.append("<h3>Why the scheduler refused</h3>")
                    P.append(table(["field", "value"],
                                   [("phase", last.get("phase")),
                                    ("waiting", (last.get("waiting") or "—")[:180])]
                                   + [("event", e[:180])
                                      for e in (last.get("events") or [])]))

        # ---- per-arm comparison ---------------------------------------
        by_role = {}
        arms_seen = []
        for sub in run["subruns"]:
            arm = sub.get("arm")
            if arm and arm not in arms_seen:
                arms_seen.append(arm)
            for role, b in ((sub.get("summary") or {}).get("per_service")
                            or {}).items():
                by_role.setdefault(role, {})[arm] = b
        arms_seen = [a for a in ["dedicated", "hami", "hami-mps"]
                     if a in arms_seen]

        if by_role:
            roles = sorted(by_role)
            P.append("<h3>Throughput</h3>")
            P.append(grouped_bars(
                "Output tokens per second, per tenant", roles, arms_seen,
                {a: {r: _num((by_role[r].get(a) or {}).get("tokens_per_second"))
                     for r in roles} for a in arms_seen},
                note="Higher is better. The dedicated bar is the same tenant "
                     "alone on the card at the same memory quota."))

            P.append("<h3>Time to first token — prefill contention</h3>")
            P.append(grouped_bars(
                "TTFT p99 (ms)", roles, arms_seen,
                {a: {r: _g(by_role[r].get(a) or {}, "ttft", "p99", 1000)
                     for r in roles} for a in arms_seen}, unit="",
                note="Lower is better. TTFT is damaged by PREFILL contention; "
                     "round one could not measure it at all because it was "
                     "not streaming."))

            P.append("<h3>Inter-token latency — decode contention</h3>")
            P.append(grouped_bars(
                "ITL p95 (ms)", roles, arms_seen,
                {a: {r: _g(by_role[r].get(a) or {}, "itl", "p95", 1000)
                     for r in roles} for a in arms_seen},
                note="Lower is better. This is what a user perceives as "
                     "stutter, and it moves independently of TTFT."))

            head = ["role", "metric"] + arms_seen + ["hami vs ded", "mps vs hami"]
            rows = []
            for r in roles:
                for name, f in ROWS:
                    vals = [_num(f(by_role[r].get(a) or {})) for a in arms_seen]
                    if all(v is None for v in vals):
                        continue
                    got = dict(zip(arms_seen, vals))
                    rows.append([r, name] + [_fmt(v) for v in vals]
                                + [_pct(got.get("hami"), got.get("dedicated")),
                                   _pct(got.get("hami-mps"), got.get("hami"))])
            P.append("<h3>Every metric, every arm</h3>")
            P.append(table(head, rows))

        # ---- time series from the saved Prometheus snapshot -------------
        for sub in run["subruns"]:
            prom, t0 = sub.get("_prom") or {}, sub.get("t0")
            org = t0 or (prom.get("t0") if isinstance(prom, dict) else None)
            figs = []
            s = prom_series(prom, "gen_tok_s", t_origin=org)
            if s:
                figs.append(lines("Generation tokens/s per tenant — %s"
                                  % sub.get("tag"), s,
                                  note="vLLM's own counter, 5 s scrape."))
            s = prom_series(prom, "waiting", t_origin=org)
            if s:
                figs.append(lines("Queue depth (waiting) — %s" % sub.get("tag"),
                                  s, note="Only observable because the client "
                                          "drives an open loop."))
            mem, util = gpu_csv(sub.get("_gpu") or "", t_origin=org)
            if mem:
                figs.append(lines("Device memory used (MiB) — %s"
                                  % sub.get("tag"),
                                  sorted(mem.items()),
                                  note="1 Hz nvidia-smi. Finer than the 5 s "
                                       "DCGM scrape, which can miss the peak."))
            if figs:
                P.append("<h3>%s over time</h3>" % _esc(sub.get("tag")))
                P.extend(figs)

    P.append('<p class="note">SM occupancy is absent by necessity, not by '
             'omission: <code>DCGM_FI_PROF_*</code> is a datacenter-GPU feature '
             'and neither exporter on this RTX 3080 Ti node publishes it. '
             '<code>DCGM_FI_DEV_GPU_UTIL</code> is not a substitute — round one '
             'found it pinned at 95–99% in nearly every configuration, so it '
             'cannot separate two tenants. Per-tenant tokens/s and queue depth '
             'do that work here.</p>')

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("".join(P))
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default="", help="where to write; default the "
                                              "first run directory")
    a = ap.parse_args()

    dirs = []
    for pat in a.run_dirs:
        dirs += sorted(glob.glob(pat)) or [pat]
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        sys.exit("no run directories matched")

    runs = collect(dirs)
    if not runs:
        sys.exit("no run.json found in: %s" % ", ".join(dirs))

    out = a.out or dirs[0]
    os.makedirs(out, exist_ok=True)
    n = write_summary_csv(runs, os.path.join(out, "summary.csv"))
    write_arms_csv(runs, os.path.join(out, "arms.csv"))
    rep = build_report(runs, os.path.join(out, "report.html"))

    print("runs        : %d" % len(runs))
    print("sub-runs    : %d" % sum(len(r["subruns"]) for r in runs))
    print("summary.csv : %d rows" % n)
    print("arms.csv    : written")
    print("report.html : %s" % rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
