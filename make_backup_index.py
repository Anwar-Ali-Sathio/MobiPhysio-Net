"""Render the recovery index page(s) for the claude.ai artifact backup from *.parts.json."""
import html
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
site = out / "site"
site.mkdir(exist_ok=True)
summary = json.loads((out / "backup_summary.json").read_text())

CSS = """
:root{--bg:#f7f8f6;--fg:#15201c;--muted:#56625c;--line:#d6ddd8;--accent:#1d6b57;--chip:#e6efe9;--mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;--sans:'IBM Plex Sans',system-ui,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#101614;--fg:#e4ebe7;--muted:#9aa8a1;--line:#2a3531;--accent:#5fc3a6;--chip:#1b2622;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#101614;--fg:#e4ebe7;--muted:#9aa8a1;--line:#2a3531;--accent:#5fc3a6;--chip:#1b2622;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--sans);line-height:1.5;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:28px 48px}
main{max-width:980px;margin:0 auto;display:grid;gap:28px}
h1{font-size:1.6rem;margin:0;text-wrap:balance}h2{font-size:1.05rem;margin:0 0 8px;letter-spacing:.02em}
p{margin:0;max-width:70ch;color:var(--muted)} .lede{color:var(--fg)}
.meta{display:flex;flex-wrap:wrap;gap:8px}.chip{background:var(--chip);border-radius:4px;padding:2px 8px;font-family:var(--mono);font-size:.78rem}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:.82rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-weight:600}
td.h{font-family:var(--mono);font-size:.74rem}
pre{margin:0;background:var(--chip);padding:12px;border-radius:6px;overflow-x:auto;font-family:var(--mono);font-size:.78rem}
ul{margin:0;padding-left:20px;color:var(--muted)} a{color:var(--accent)}
"""


def page(pkgs, title, intro, extra=""):
    secs = []
    for p in pkgs:
        rows = "".join(f"<tr><td><a href=\"{html.escape(x['file'])}\">{html.escape(x['file'])}</a></td><td>{x['bytes']:,}</td>"
                       f"<td class=h>{x['sha256']}</td></tr>" for x in p["parts"])
        secs.append(f"""<section><h2>{html.escape(p['archive'])}</h2>
<div class=meta><span class=chip>{p['archive_bytes']:,} bytes</span><span class=chip>{p['n_files']} files inside</span><span class=chip>{len(p['parts'])} parts</span></div>
<p style="margin-top:8px">Archive SHA-256 <span class=chip>{p['archive_sha256']}</span></p>
<div class=wrap style="margin-top:10px"><table><thead><tr><th>Part</th><th>Bytes</th><th>SHA-256</th></tr></thead><tbody>{rows}</tbody></table></div>
<pre style="margin-top:10px">sha256sum -c --ignore-missing parts.sha256   # parts now; archive line again after cat
{html.escape(p['reassemble'])}
unzip -t {html.escape(p['archive'])}</pre></section>""")
    return f"""<title>{html.escape(title)}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>{CSS}</style>
<main><header style="display:grid;gap:8px"><h1>{html.escape(title)}</h1><p class=lede>{intro}</p>
<div class=meta><span class=chip>snapshot {summary['created_utc']}</span><span class=chip>experiment full_v2</span></div></header>
{''.join(secs)}{extra}</main>"""


p1, p2 = summary["packages"]
nb = "".join(f"<li><b>{html.escape(k)}</b>: {html.escape(v)}</li>" for k, v in summary["not_backed_up"].items())
extra1 = f"""<section><h2>Restore</h2><p>Download every part (or ask Claude to read this artifact's files), verify, reassemble, then unzip into a new directory. The archive holds <code>MobiPhysio-Net/</code> (code, configs, data manifests, every run record and per-video prediction, results, reports, logs, historical archive) and <code>data_dir/</code> (checkpoints of completed runs, decode provenance, official README and annotation CSV). Start with <code>MobiPhysio-Net/reports/next_steps_after_reconnection.md</code> and <code>reports/run_status.md</code>.</p></section>
<section><h2>Not in this backup</h2><ul>{nb}</ul><p style="margin-top:8px">Large files (fine-tuned-ViT checkpoints, ViT and SVM feature caches) are in the companion artifact "MobiPhysio-Net Large Files".</p></section>"""
(site / "recovery.html").write_text(page([p1], "MobiPhysio-Net Recovery Backup",
    "Snapshot of the MobiPhysio-Net project taken while the remaining seed runs were still training. Split into 14 MB parts so each part fits the artifact file limit.", extra1))
(site / "large.html").write_text(page([p2], "MobiPhysio-Net Large Files",
    "Companion to the recovery backup: the fine-tuned-ViT checkpoint(s) of completed runs and the ViT-S/16 and SVM feature caches (stored, not compressed). Restore into <code>$MOBIPHYSIO_DATA_DIR</code> after unzipping."))
for p, n in [(p1, "recovery"), (p2, "large")]:
    (site / f"parts_{n}.sha256").write_text("".join(f"{x['sha256']}  {x['file']}\n" for x in p["parts"]) + f"{p['archive_sha256']}  {p['archive']}\n")
print("ok")
