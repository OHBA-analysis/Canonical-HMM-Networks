"""Generate a QC summary HTML report from pipeline plots."""

import base64
import html
from pathlib import Path
from datetime import datetime

STEPS = {
    1: {
        "name": "Preprocessing",
        "files": [
            "1_sum_square.png",
            "1_sum_square_exclude_bads.png",
            "1_channel_stds.png",
        ],
    },
    2: {
        "name": "Surfaces",
        "files": [
            "2_inskull.png",
            "2_outskin.png",
            "2_outskull.png",
        ],
    },
    3: {
        "name": "Coregistration",
        "files": [
            "3_coreg.html",
        ],
    },
    4: {
        "name": "Source Recon & Parcellation",
        "files": [
            "4_psd_topo.png",
        ],
    },
}

CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0;
    padding: 20px;
    background: #f5f5f5;
    color: #333;
}
h1 {
    margin: 0 0 20px 0;
}
.tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 20px;
    border-bottom: 2px solid #ddd;
}
.tab-btn {
    padding: 10px 20px;
    border: none;
    background: #e0e0e0;
    cursor: pointer;
    font-size: 14px;
    border-radius: 6px 6px 0 0;
    transition: background 0.2s;
}
.tab-btn:hover {
    background: #d0d0d0;
}
.tab-btn.active {
    background: #fff;
    font-weight: bold;
    border-bottom: 2px solid #fff;
    margin-bottom: -2px;
}
.tab-content {
    display: none;
    background: #fff;
    padding: 20px;
    border-radius: 0 0 6px 6px;
}
.tab-content.active {
    display: block;
}
.session-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(450px, 1fr));
    gap: 20px;
}
.session-card {
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 15px;
    background: #fafafa;
}
.session-card h3 {
    margin: 0 0 10px 0;
    font-size: 15px;
    color: #555;
}
.session-card img {
    max-width: 100%;
    max-height: 400px;
    display: block;
    margin: 8px 0;
    border: 1px solid #eee;
}
.session-card iframe {
    width: 100%;
    height: 500px;
    border: 1px solid #ddd;
    border-radius: 4px;
    margin: 8px 0;
}
.placeholder {
    background: #eee;
    color: #999;
    padding: 40px;
    text-align: center;
    border-radius: 4px;
    margin: 8px 0;
    font-style: italic;
}
.file-label {
    font-size: 12px;
    color: #888;
    margin: 12px 0 2px 0;
}
.footer {
    margin-top: 30px;
    padding-top: 15px;
    border-top: 1px solid #ddd;
    font-size: 12px;
    color: #999;
}
"""

JS = """
function switchTab(step) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById('btn-' + step).classList.add('active');
    document.getElementById('tab-' + step).classList.add('active');
}
"""


def _embed_png(filepath):
    """Encode a PNG file as a base64 data URI."""
    data = filepath.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}">'


def _embed_html(filepath):
    """Embed an HTML file as an iframe with srcdoc."""
    content = filepath.read_text(errors="replace")
    escaped = html.escape(content, quote=True)
    return f'<iframe srcdoc="{escaped}" loading="lazy"></iframe>'


def _build_step_tab(step_num, step_info, plots_dir, sessions):
    """Build the HTML content for a single step tab."""
    files = step_info["files"]

    # Count how many sessions have at least one file for this step
    done = 0
    for session_id in sessions:
        session_dir = plots_dir / session_id
        if any((session_dir / f).exists() for f in files):
            done += 1
    total = len(sessions)

    parts = []
    parts.append(f'<div class="session-grid">')

    for session_id in sessions:
        session_dir = plots_dir / session_id
        parts.append(f'<div class="session-card">')
        parts.append(f"<h3>{session_id}</h3>")

        for filename in files:
            filepath = session_dir / filename
            parts.append(f'<div class="file-label">{filename}</div>')
            if filepath.exists():
                if filename.endswith(".png"):
                    parts.append(_embed_png(filepath))
                elif filename.endswith(".html"):
                    parts.append(_embed_html(filepath))
            else:
                parts.append('<div class="placeholder">Pending</div>')

        parts.append("</div>")

    parts.append("</div>")
    return "\n".join(parts), done, total


def generate_report(plots_dir, sessions, output_file="report.html"):
    """Generate a QC summary HTML report.

    Scans the plots directory for existing QC files and builds a
    self-contained HTML report with tabs for each pipeline step.

    Parameters
    ----------
    plots_dir : str or Path
        Path to the plots directory containing per-session subdirectories.
    sessions : dict
        Dictionary of sessions (same format as the pipeline scripts).
    output_file : str, optional
        Filename for the report. Written to plots_dir/output_file.
    """
    plots_dir = Path(plots_dir)

    # Build tab buttons and content
    tab_buttons = []
    tab_contents = []

    first = True
    for step_num, step_info in STEPS.items():
        content, done, total = _build_step_tab(
            step_num, step_info, plots_dir, sessions
        )
        active = " active" if first else ""
        tab_buttons.append(
            f'<button class="tab-btn{active}" id="btn-{step_num}" '
            f'onclick="switchTab({step_num})">'
            f'Step {step_num}: {step_info["name"]} ({done}/{total})'
            f"</button>"
        )
        tab_contents.append(
            f'<div class="tab-content{active}" id="tab-{step_num}">'
            f"{content}</div>"
        )
        first = False

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>QC Report</title>
<style>{CSS}</style>
</head>
<body>
<h1>QC Report</h1>
<div class="tabs">
{"".join(tab_buttons)}
</div>
{"".join(tab_contents)}
<div class="footer">Generated: {timestamp}</div>
<script>{JS}</script>
</body>
</html>"""

    output_path = plots_dir / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Report saved: {output_path}")
