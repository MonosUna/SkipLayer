import html
import re


def _anchor_id(name: str) -> str:
    anchor = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()
    return anchor or "html-report"


def _build_metric_section(name: str, step_to_html: dict[int, str]) -> str:
    steps = sorted(step_to_html.keys(), reverse=True)
    if not steps:
        return ""

    latest_step = steps[0]
    anchor = _anchor_id(name)
    escaped_name = html.escape(name)
    step_sections = []
    for step in steps:
        open_attr = " open" if step == latest_step else ""
        step_sections.append(
            f"""
      <details class="step-block"{open_attr}>
        <summary>Step {step}</summary>
        <div class="step-content">{step_to_html[step]}</div>
      </details>"""
        )

    return f"""
    <section class="metric-section" id="{anchor}">
      <h2>{escaped_name}</h2>
      <p class="metric-meta">{len(steps)} logged step(s). Latest: {latest_step}.</p>
      {''.join(step_sections)}
    </section>"""


def build_html_dashboard(html_by_key: dict[str, dict[int, str]]) -> str:
    sections = []
    nav_links = []
    for key in sorted(html_by_key.keys()):
        section = _build_metric_section(key, html_by_key[key])
        if not section:
            continue
        sections.append(section)
        nav_links.append(f'<a href="#{_anchor_id(key)}">{html.escape(key)}</a>')

    if not sections:
        return "<p>No data yet.</p>"

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>SkipLayer HTML report</title>
  <style>
    body {{ font-family: sans-serif; margin: 16px; line-height: 1.4; }}
    .page-title {{ margin: 0 0 8px; }}
    .page-subtitle {{ color: #555; margin: 0 0 14px; }}
    .toc {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; }}
    .toc a {{ color: #0b57d0; text-decoration: none; border: 1px solid #d0d7de; border-radius: 999px; padding: 4px 10px; }}
    .toc a:hover {{ text-decoration: underline; }}
    .metric-section {{ margin: 24px 0; padding-top: 8px; border-top: 1px solid #e5e7eb; }}
    .metric-meta {{ color: #555; margin: 0 0 10px; }}
    .step-block {{ margin: 10px 0; border: 1px solid #ddd; border-radius: 6px; padding: 8px 10px; }}
    .step-block > summary {{ cursor: pointer; font-weight: 600; }}
    .step-content {{ margin-top: 10px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1 class="page-title">HTML report</h1>
  <p class="page-subtitle">Static rendering for Comet HTML logs.</p>
  <nav class="toc">{''.join(nav_links)}</nav>
  {''.join(sections)}
</body>
</html>"""
