"""Concept Skill Tagger — Streamlit UI."""
from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import tomllib
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import classroom_content

APP_DIR = Path(__file__).resolve().parent
SECRETS_PATH = APP_DIR / ".streamlit" / "secrets.toml"

st.set_page_config(
    page_title="Concept Skill Tagger",
    layout="wide",
)


def _secrets_from_toml() -> dict[str, str]:
    """Load secrets from app/.streamlit/secrets.toml (path-stable regardless of cwd)."""
    if not SECRETS_PATH.is_file():
        return {}
    with SECRETS_PATH.open("rb") as f:
        data = tomllib.load(f)
    return {k: str(v).strip() for k, v in data.items() if isinstance(v, str)}


def _app_password() -> str:
    try:
        value = str(st.secrets["APP_PASSWORD"]).strip()
        if value:
            return value
    except (KeyError, FileNotFoundError, AttributeError, TypeError):
        pass
    return _secrets_from_toml().get("APP_PASSWORD", "").strip()


def _check_password() -> bool:
    """Return True once the shared app password has been entered this session."""
    if st.session_state.get("authenticated"):
        return True

    expected = _app_password()
    if not expected:
        st.error(
            f"Set `APP_PASSWORD` in `{SECRETS_PATH}`. "
            "The password is local-only and is not stored in the repo."
        )
        return False

    st.title("Concept Skill Tagger")
    st.caption("Enter the password to continue.")
    with st.form("login"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        if password and hmac.compare_digest(password, expected):
            st.session_state.authenticated = True
            st.rerun()
        st.error("Incorrect password.")
    return False


if not _check_password():
    st.stop()


def _load_secrets() -> tuple[str, str]:
    jwt = ""
    api_key = ""
    try:
        jwt = str(st.secrets["UDACITY_JWT"]).strip()
        api_key = str(st.secrets["OPENAI_API_KEY"]).strip()
    except (KeyError, FileNotFoundError, AttributeError, TypeError):
        pass

    local = _secrets_from_toml()
    jwt = jwt or local.get("UDACITY_JWT", "").strip()
    api_key = api_key or local.get("OPENAI_API_KEY", "").strip()

    if jwt and api_key:
        return jwt, api_key

    if SECRETS_PATH.is_file():
        st.error(
            f"Secrets in `{SECRETS_PATH}` are empty on disk. "
            "If you edited the file in your IDE, save it (Cmd+S) and refresh this page."
        )
    else:
        st.error(
            f"Missing `{SECRETS_PATH}`. Copy secrets.toml.example and set "
            "UDACITY_JWT and OPENAI_API_KEY."
        )
    st.stop()
    return "", ""  # unreachable; satisfies type checkers


if "program_meta" not in st.session_state:
    st.session_state.program_meta = None
if "workspace_concepts" not in st.session_state:
    st.session_state.workspace_concepts = []
if "results" not in st.session_state:
    st.session_state.results = []
if "analyze_settings" not in st.session_state:
    st.session_state.analyze_settings = {
        "consensus_enabled": True,
        "n_runs": 3,
    }


def _classroom_url(program_key: str, concept_key: str) -> str:
    return f"https://classroom.udacity.com/{program_key}?conceptKey={concept_key}"


_DATA_FLOW_MERMAID = """
flowchart TD
    subgraph input ["Input"]
        KEY["cd / nd key"]
        SECRETS["UDACITY_JWT + OPENAI_API_KEY"]
    end

    subgraph resolve ["1. Resolve program"]
        CC["classroom-content GraphQL"]
        META["Program metadata<br/>teaches_skills allowlist"]
        KEY --> CC
        SECRETS --> CC
        CC --> META
    end

    subgraph extract ["2. Build concept context"]
        SCAN["Find concepts with WorkspaceAtom"]
        FILTER["Keep concepts owned by this component;\nskip nested child components\n(e.g. ls inside cd)"]
        ATOMS["Text, Video VTT, Quiz atoms"]
        WS["WorkspaceAtom metadata"]
        PROV["workspace-provisioner<br/>/masterfiles/download"]
        GCS["GCS udacity-masterfiles<br/>starter .tar.gz"]
        SCOPE["Scope to main_default_path<br/>exercise folder + ancestor READMEs"]
        CTX["Concept context text"]
        SKIP["Skip note: in child component"]
        CC --> SCAN
        SCAN --> FILTER
        FILTER --> ATOMS
        FILTER --> WS
        FILTER --> SKIP
        ATOMS --> CTX
        WS --> PROV
        SECRETS --> PROV
        PROV --> GCS
        GCS --> SCOPE
        WS --> SCOPE
        SCOPE --> CTX
    end

    subgraph tag ["3. Recommend skills"]
        PROMPT["Prompt: allowlist + context<br/>learner practice / exposure"]
        LLM["OpenAI structured output<br/>1-3 skills + rationale"]
        VALID["Validate names against allowlist"]
        CONS["Consensus majority vote<br/>optional N runs"]
        META --> PROMPT
        CTX --> PROMPT
        PROMPT --> LLM
        LLM --> VALID
        VALID --> CONS
    end

    subgraph output ["Output"]
        UI["Table or Cards view"]
        CSV["Download CSV"]
        CONS --> UI
        CONS --> CSV
        SKIP --> UI
        SKIP --> CSV
    end
"""


def _render_mermaid_dark(diagram: str, *, height: int = 720) -> None:
    """Render a Mermaid diagram with Cursor-like dark styling."""
    diagram_text = diagram.strip()
    graph_id = f"mermaid-{hashlib.sha256(diagram_text.encode()).hexdigest()[:10]}"
    diagram_js = json.dumps(diagram_text)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  html, body {{
    margin: 0;
    padding: 10px 14px;
    background: #1e1e1e;
    color: #d4d4d4;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  #graph-container {{
    display: flex;
    justify-content: center;
    min-height: 200px;
  }}
  #graph-container svg {{
    max-width: 100%;
    height: auto;
  }}
  #graph-error {{
    color: #f48771;
    font-size: 13px;
    padding: 8px 0;
  }}
</style>
</head>
<body>
<div id="graph-container"></div>
<div id="graph-error" hidden></div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
  const diagram = {diagram_js};
  const graphId = "{graph_id}";
  let rendered = false;

  mermaid.initialize({{
    startOnLoad: false,
    securityLevel: "loose",
    theme: "base",
    themeVariables: {{
      darkMode: true,
      background: "#1e1e1e",
      primaryColor: "#2d2d30",
      primaryTextColor: "#d4d4d4",
      primaryBorderColor: "#569cd6",
      secondaryColor: "#252526",
      secondaryTextColor: "#d4d4d4",
      secondaryBorderColor: "#454545",
      tertiaryColor: "#1e1e1e",
      tertiaryTextColor: "#cccccc",
      tertiaryBorderColor: "#454545",
      lineColor: "#6e7681",
      textColor: "#d4d4d4",
      mainBkg: "#2d2d30",
      nodeBorder: "#569cd6",
      clusterBkg: "#252526",
      clusterBorder: "#454545",
      titleColor: "#cccccc",
      edgeLabelBackground: "#252526",
      fontFamily: "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
      fontSize: "14px"
    }},
    flowchart: {{ htmlLabels: true, curve: "basis", padding: 16 }}
  }});

  async function draw() {{
    if (rendered) return;
    const container = document.getElementById("graph-container");
    const errEl = document.getElementById("graph-error");
    if (!container) return;
    // Wait until iframe is visible (expander open)
    if (container.offsetWidth === 0 && container.offsetHeight === 0) return;
    try {{
      const {{ svg }} = await mermaid.render(graphId, diagram);
      container.innerHTML = svg;
      rendered = true;
      if (errEl) errEl.hidden = true;
    }} catch (err) {{
      console.error("mermaid render failed", err);
      if (errEl) {{
        errEl.textContent = "Diagram failed to render. Refresh the page.";
        errEl.hidden = false;
      }}
    }}
  }}

  function scheduleDraw() {{
    requestAnimationFrame(draw);
  }}

  scheduleDraw();
  const observer = new IntersectionObserver((entries) => {{
    if (entries.some((e) => e.isIntersecting)) scheduleDraw();
  }}, {{ threshold: 0.01 }});
  observer.observe(document.body);
  window.addEventListener("load", scheduleDraw);
  setTimeout(scheduleDraw, 300);
</script>
</body>
</html>"""
    components.html(html, height=height, scrolling=True)


def _render_data_flow_expander() -> None:
    with st.expander("How it works", expanded=False):
        _render_mermaid_dark(_DATA_FLOW_MERMAID, height=800)


def _results_to_rows(results: list[dict[str, Any]], n_runs: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in results:
        row: dict[str, Any] = {
            "concept_title": r.get("concept_title", ""),
            "concept_key": r.get("concept_key", ""),
            "lesson_title": r.get("lesson_title", ""),
            "skipped": "yes" if r.get("skipped") else "no",
            "skip_reason": r.get("skip_reason", ""),
            "child_component_key": r.get("child_component_key", ""),
            "consensus_skills": "; ".join(r.get("consensus_skills") or []),
            "agreement_score": r.get("agreement_score", 0),
            "validation_status": r.get("validation_status", ""),
            "workspace_files_extracted": "yes" if r.get("workspace_files_included") else "no",
            "workspace_files_chars": r.get("workspace_files_chars", 0),
            "workspace_scope": r.get("workspace_scope_status", ""),
            "workspace_scope_paths": "; ".join(r.get("workspace_scope_paths") or []),
            "llm_input_text": r.get("llm_input_text", ""),
        }
        run_skills = r.get("run_skills") or []
        for i in range(n_runs):
            skills = run_skills[i] if i < len(run_skills) else []
            row[f"run_{i + 1}_skills"] = "; ".join(skills)
        rows.append(row)
    return rows


def _results_to_csv(results: list[dict[str, Any]], n_runs: int) -> str:
    rows = _results_to_rows(results, n_runs)
    if not rows:
        return ""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def _results_as_csv_dataframe(results: list[dict[str, Any]], n_runs: int) -> pd.DataFrame:
    """DataFrame built from the same CSV bytes as Download CSV."""
    csv_text = _results_to_csv(results, n_runs)
    if not csv_text:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(csv_text))


def _display_n_runs(results: list[dict[str, Any]], settings: dict[str, Any]) -> int:
    from_runs = max(len(r.get("run_skills") or []) for r in results) if results else 1
    if settings.get("consensus_enabled"):
        return max(settings.get("n_runs", 3), from_runs)
    return max(1, from_runs)


def _render_skill_pills(skills: list[str], *, label: str = "Skills:") -> None:
    st.markdown(f"**{label}**")
    if not skills:
        st.caption("—")
        return
    with st.container(horizontal=True, gap="small", horizontal_alignment="left"):
        for skill in skills:
            st.badge(skill, color="blue")


def _render_card(r: dict[str, Any], program_key: str) -> None:
    title = r.get("concept_title", "")
    ck = r.get("concept_key", "")
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(ck)
        if program_key and ck:
            st.markdown(f"[Open in Classroom]({_classroom_url(program_key, ck)})")
        if r.get("skipped"):
            child_key = r.get("child_component_key") or ""
            st.info(r.get("skip_reason") or "Skill tagging skipped because this concept is in a child component.")
            if child_key:
                st.caption(f"Tag this concept by analyzing `{child_key}` instead.")
            return
        _render_skill_pills(r.get("consensus_skills") or [])
        if r.get("rationale"):
            st.markdown(f"**Rationale:** {r.get('rationale')}")
        if r.get("workspace_files_included"):
            st.caption("✅ starter files found")
            scope_paths = r.get("workspace_scope_paths") or []
            status = r.get("workspace_scope_status", "")
            if scope_paths:
                st.caption(f"📁 exercise folder: `{'; '.join(scope_paths)}`")
            elif status == "unresolved":
                st.caption("📁 default path did not match the archive — whole workspace used")
            elif status == "full_workspace":
                st.caption("📁 no per-concept default path — whole workspace used")
        else:
            st.caption("❌ starter files not found")
        if r.get("validation_errors"):
            st.warning("; ".join(r.get("validation_errors") or []))
        with st.expander("LLM input"):
            st.code(r.get("llm_input_text", "") or "", language=None)


def _render_cards_grid(results: list[dict[str, Any]], program_key: str) -> None:
    for batch_start in range(0, len(results), 3):
        batch = results[batch_start:batch_start + 3]
        cols = st.columns(3, gap="medium")
        for i, r in enumerate(batch):
            with cols[i]:
                _render_card(r, program_key)


def _clear_progress(
    header: st.delta_generator.DeltaGenerator,
    bar: st.delta_generator.DeltaGenerator,
    detail: st.delta_generator.DeltaGenerator,
) -> None:
    header.empty()
    bar.empty()
    detail.empty()


def _run_analyze(
    program_key: str,
    *,
    consensus_enabled: bool,
    n_runs: int,
    include_prerequisites: bool,
    progress_header: st.delta_generator.DeltaGenerator,
    progress_bar: st.delta_generator.DeltaGenerator,
    progress_detail: st.delta_generator.DeltaGenerator,
) -> None:
    jwt, api_key = _load_secrets()

    def on_progress(msg: str, current: int = 0, total: int = 0) -> None:
        if total > 0:
            progress_header.markdown(f"**Analyzing program — {current}/{total}**")
            progress_bar.progress(min(current / total, 1.0))
        else:
            progress_header.markdown("**Analyzing program**")
            progress_bar.progress(0.0)
        progress_detail.caption(msg)

    try:
        outcome = classroom_content.analyze_program(
            program_key,
            jwt,
            api_key,
            include_prerequisites=include_prerequisites,
            consensus_enabled=consensus_enabled,
            n_runs=n_runs if consensus_enabled else 1,
            progress_callback=on_progress,
        )
        st.session_state.program_meta = outcome["program_meta"]
        st.session_state.workspace_concepts = outcome["workspace_concepts"]
        st.session_state.results = outcome["results"]
        st.session_state.analyze_settings = {
            "consensus_enabled": consensus_enabled,
            "n_runs": n_runs,
        }
        if not outcome["workspace_concepts"]:
            progress_detail.caption("No concepts with WorkspaceAtoms found in this program.")
    except classroom_content.UdacityAPIError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    finally:
        _clear_progress(progress_header, progress_bar, progress_detail)


# ---- Main (header + progress before sidebar so updates render in main) ----

st.title("Concept Skill Tagger")
st.markdown(
    "Enter a **cd** or **nd** key, find concepts with workspaces, and recommend "
    "1–3 program-level skills per concept. Only concepts **owned by this component** "
    "are tagged; concepts in a nested child component (for example an **ls** lesson "
    "library inside a **cd**) are listed with a skip note."
)

_render_data_flow_expander()

progress_header = st.empty()
progress_bar = st.empty()
progress_detail = st.empty()

# ---- Sidebar ----

with st.sidebar:
    st.header("Concept Skill Tagger")
    st.caption("Tag workspace concepts with program-level skills via OpenAI.")

    program_key = st.text_input("cd / nd key", placeholder="e.g. nd006, cd1827")
    consensus_enabled = st.toggle("Consensus mode", value=True)
    n_runs = st.slider(
        "Consensus runs",
        min_value=2,
        max_value=5,
        value=3,
        disabled=not consensus_enabled,
    )
    include_prerequisites = st.toggle("Include prerequisite skills in allowlist", value=False)

    if st.button("Analyze", type="primary", use_container_width=True):
        key = program_key.strip()
        if not key:
            st.warning("Enter a cd/nd key.")
        else:
            _run_analyze(
                key,
                consensus_enabled=consensus_enabled,
                n_runs=n_runs,
                include_prerequisites=include_prerequisites,
                progress_header=progress_header,
                progress_bar=progress_bar,
                progress_detail=progress_detail,
            )

    if st.session_state.program_meta:
        meta = st.session_state.program_meta
        st.divider()
        st.caption(f"**{meta.get('title')}** (`{meta.get('key')}`)")
        if meta.get("unreleased"):
            st.caption("Unreleased / construction branch")
        st.caption(f"Allowed skills: {len(meta.get('allowed_skills') or [])}")
        tagged = meta.get("tagged_concept_count")
        skipped = meta.get("skipped_child_concept_count", 0)
        if tagged is None:
            st.caption(f"Workspace concepts: {meta.get('workspace_concept_count', 0)}")
        else:
            st.caption(
                f"Workspace concepts: {meta.get('workspace_concept_count', 0)} "
                f"({tagged} tagged, {skipped} skipped — child component)"
            )

# ---- Results ----

results = st.session_state.results
meta = st.session_state.program_meta
settings = st.session_state.analyze_settings

if meta and not results and meta.get("workspace_concept_count", 0) == 0:
    st.warning(
        f"Program **{meta.get('title')}** (`{meta.get('key')}`) loaded but no concepts "
        "with WorkspaceAtoms were found."
    )
    st.stop()

if not results:
    st.info("Load a program from the sidebar to see tagging results.")
    st.stop()

program_key = (meta or {}).get("key", "")
display_n = _display_n_runs(results, settings)
csv_data = _results_to_csv(results, display_n)

st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name=f"{program_key}_skill_tags.csv",
    mime="text/csv",
)

view_mode = st.radio("View", ["Table", "Cards"], horizontal=True, index=0, key="results_view")

teaches_skills = (meta or {}).get("teaches_skills") or []
_render_skill_pills(teaches_skills, label="Teaches skills:")

st.markdown(
    "Skills in **consensus** appeared in a majority of LLM runs. "
    "Low-confidence skills appeared in only one run. "
    "Concepts in a nested child component are **not tagged** here — analyze that "
    "child component's key to tag them."
)

if view_mode == "Table":
    st.subheader("CSV preview")
    st.caption("Same columns and values as Download CSV.")
    st.dataframe(
        _results_as_csv_dataframe(results, display_n),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.subheader("Cards")
    _render_cards_grid(results, program_key)
