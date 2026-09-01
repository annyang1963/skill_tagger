"""Udacity classroom-content client, workspace concept extraction, and skill tagging via OpenAI."""
from __future__ import annotations

import io
import json
import math
import re
import tarfile
from collections import Counter
from pathlib import PurePosixPath
from typing import Any

import requests
from openai import OpenAI
from pydantic import BaseModel, Field

CLASSROOM_CONTENT_GRAPHQL = "https://api.udacity.com/api/classroom-content/v1/graphql"
WORKSPACE_PROVISIONER_BASE = "https://api.udacity.com/api/workspace-provisioner"
DEFAULT_LOCALE = "en-us"
_TIMEOUT = 60
ND_KEY_PATTERN = re.compile(r"^nd", re.IGNORECASE)
DEFAULT_MODEL = "gpt-4o-mini"
VTT_MAX_CHARS = 2000
RATIONALE_MAX_LENGTH = 255
WORKSPACE_FILES_MAX_CHARS = 20000
WORKSPACE_FILE_MAX_BYTES = 100_000
TEXT_EXTENSIONS = frozenset({
    ".py", ".ipynb", ".md", ".txt", ".json", ".html", ".css", ".js",
    ".sh", ".yaml", ".yml", ".sql", ".r", ".xml", ".toml", ".ini", ".cfg",
})
SKIP_ARCHIVE_DIRS = frozenset({"node_modules", ".git", "__pycache__", ".venv", "venv"})

QUERIES = """
query ComponentsByKey($key: String!) {
  components(key: $key, count: 50) {
    id
    key
    locale
    type
    deprecated
    latest_release {
      root_node_id
    }
  }
}

query ComponentByKey($key: String!, $locale: String!) {
  component(key: $key, locale: $locale) {
    latest_release {
      major
      minor
      patch
      root_node_id
      root_node {
        id
        key
        locale
        version
        title
      }
      component {
        metadata {
          difficulty_level { name uri }
          teaches_skills { name uri }
          prerequisite_skills { name uri }
        }
      }
    }
  }
}

query ConstructionByKey($key: String!, $locale: String!) {
  node(key: $key, locale: $locale, version: "construction") {
    id
    key
    title
    semantic_type
  }
  component(key: $key, locale: $locale) {
    metadata {
      difficulty_level { name uri }
      teaches_skills { name uri }
      prerequisite_skills { name uri }
    }
  }
}

fragment conceptFields on Concept {
  id
  key
  title
  is_public
  progress_key
  atoms {
    __typename
    ... on AtomInterface {
      id
      key
      semantic_type
      title
    }
    ... on TextAtom {
      text
    }
    ... on VideoAtom {
      video { vtt_url }
    }
    ... on WorkspaceAtom {
      id
      key
      branch_id
      semantic_type
      title
      name
      instructor_notes
      workspace_id
      pool_id
      master_archive_id
      main_default_path
      configuration
    }
    ... on RadioQuizAtom {
      question {
        prompt
        answers { is_correct text }
      }
    }
    ... on CheckboxQuizAtom {
      question {
        prompt
        correct_feedback
        answers { is_correct text }
      }
    }
    ... on MatchingQuizAtom {
      question {
        answers_label
        concepts_label
        concepts { text correct_answer { text } }
        complex_prompt { text }
        answers { text }
      }
    }
  }
}

fragment lessonFields on Lesson {
  id
  key
  title
  summary
  is_project_lesson
  concepts { ...conceptFields }
}

fragment moduleFields on Module {
  id
  key
  title
  lessons { ...lessonFields }
}

fragment partFields on Part {
  id
  key
  title
  summary
  is_optional
  is_public
  modules { ...moduleFields }
}

query NodeById($id: Int!) {
  node(id: $id) {
    id
    key
    title
    locale
    version
    semantic_type
    ... on Nanodegree {
      summary
      syllabus_overview
      parts { ...partFields }
    }
    ... on Part {
      ...partFields
    }
    ... on Lesson {
      ...lessonFields
    }
  }
}
"""


class UdacityAPIError(RuntimeError):
    pass


class SkillRecommendation(BaseModel):
    recommended_skills: list[str] = Field(
        description=(
            "1 to 3 skill names from the allowed list that the learner will practice, "
            "apply, or be meaningfully exposed to by completing this concept"
        )
    )
    rationale: str = Field(
        description=(
            "One brief sentence (max 255 characters) on what the learner gains from this "
            "concept. Refer to the analyzed item as 'concept' — never 'lesson' or 'course'."
        )
    )


def is_nd_key(key: str) -> bool:
    return bool(ND_KEY_PATTERN.match((key or "").strip()))


def _auth_headers(jwt: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {jwt}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _gql(jwt: str, operation: str, variables: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(
        CLASSROOM_CONTENT_GRAPHQL,
        headers=_auth_headers(jwt),
        json={"query": QUERIES, "operationName": operation, "variables": variables},
        timeout=_TIMEOUT,
    )
    if resp.status_code in (401, 403):
        raise UdacityAPIError(
            f"classroom-content HTTP {resp.status_code}: staff JWT invalid/expired/revoked. "
            f"Refresh UDACITY_JWT. Preview: {resp.text[:200]!r}"
        )
    if not resp.ok:
        raise UdacityAPIError(f"classroom-content HTTP {resp.status_code}: {resp.text[:500]}")
    body = resp.json()
    if body.get("errors"):
        raise UdacityAPIError(f"classroom-content GraphQL errors: {body['errors']}")
    return body.get("data") or {}


def _components_by_key(jwt: str, key: str) -> list[dict[str, Any]]:
    return _gql(jwt, "ComponentsByKey", {"key": key}).get("components") or []


def _component_release(jwt: str, key: str, locale: str) -> dict[str, Any] | None:
    component = _gql(jwt, "ComponentByKey", {"key": key, "locale": locale}).get("component")
    if not component:
        return None
    release = component.get("latest_release")
    if not release or not release.get("root_node_id"):
        return None
    return release


def _construction_release(jwt: str, key: str, locale: str) -> dict[str, Any] | None:
    data = _gql(jwt, "ConstructionByKey", {"key": key, "locale": locale})
    node = data.get("node")
    if not node or not node.get("id"):
        return None
    return {
        "root_node_id": node.get("id"),
        "root_node": {"id": node.get("id"), "title": node.get("title")},
        "component": {"metadata": (data.get("component") or {}).get("metadata")},
        "_unreleased": True,
    }


def _pick_nd_locale(components: list[dict[str, Any]], requested: str = DEFAULT_LOCALE) -> str | None:
    if not components:
        return None
    available = [c.get("locale") for c in components if c.get("locale")]
    available_set = set(available)
    non_deprecated = [c.get("locale") for c in components if c.get("locale") and not c.get("deprecated")]
    if requested in available_set:
        return requested
    if DEFAULT_LOCALE in available_set:
        return DEFAULT_LOCALE
    if non_deprecated:
        return non_deprecated[0]
    return available[0]


def _pick_released_locale(components: list[dict[str, Any]]) -> str | None:
    if not components:
        return None
    has_release = lambda c: bool((c.get("latest_release") or {}).get("root_node_id"))
    pool = [c for c in components if has_release()] or components
    chosen = (
        next((c for c in pool if c.get("locale") == DEFAULT_LOCALE), None)
        or next((c for c in pool if not c.get("deprecated")), None)
        or pool[0]
    )
    return chosen.get("locale")


def _root_id_from(release: dict[str, Any]) -> int | None:
    rid = release.get("root_node_id") or (release.get("root_node") or {}).get("id")
    return int(rid) if rid is not None else None


def _resolve_root_node_id(jwt: str, key: str) -> tuple[int, str, dict[str, Any]]:
    """Return (root_node_id, locale, release dict with metadata when available)."""
    if is_nd_key(key):
        components = _components_by_key(jwt, key)
        if not components:
            raise UdacityAPIError(
                f"components(key:{key!r}) returned 0 rows — key missing or JWT lacks visibility."
            )
        locale = _pick_nd_locale(components)
        if not locale:
            raise UdacityAPIError(f"No locale found for ND key {key!r}.")
        release = _component_release(jwt, key, locale)
        if not release:
            release = _construction_release(jwt, key, locale)
        if not release:
            raise UdacityAPIError(f"ND key {key!r}: no release or CONSTRUCTION branch in locale {locale!r}.")
        root_id = _root_id_from(release)
        if root_id is None:
            raise UdacityAPIError(f"ND key {key!r}: release has no root_node_id.")
        return root_id, locale, release

    release = _component_release(jwt, key, DEFAULT_LOCALE)
    if release:
        return int(_root_id_from(release)), DEFAULT_LOCALE, release

    components = _components_by_key(jwt, key)
    locale = _pick_released_locale(components)
    if not locale or locale == DEFAULT_LOCALE:
        release = _construction_release(jwt, key, DEFAULT_LOCALE)
        if release and _root_id_from(release) is not None:
            return int(_root_id_from(release)), DEFAULT_LOCALE, release
        raise UdacityAPIError(
            f"No published release found for cd key {key!r} in any locale."
        )

    release = _component_release(jwt, key, locale)
    if not release or _root_id_from(release) is None:
        raise UdacityAPIError(f"cd key {key!r} locale {locale!r} has no release.")
    return int(_root_id_from(release)), locale, release


def _metadata_from_release(release: dict[str, Any]) -> dict[str, Any] | None:
    comp = release.get("component") or {}
    return comp.get("metadata")


def _skills_from_metadata(
    metadata: dict[str, Any] | None,
    *,
    include_prerequisites: bool = False,
) -> list[str]:
    if not metadata:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for field in ("teaches_skills", "prerequisite_skills" if include_prerequisites else None):
        if not field:
            continue
        for skill in metadata.get(field) or []:
            name = (skill or {}).get("name")
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def fetch_program_tree(jwt: str, root_node_id: int) -> dict[str, Any]:
    node = _gql(jwt, "NodeById", {"id": root_node_id}).get("node")
    if not node:
        raise UdacityAPIError(f"node(id:{root_node_id}) returned null.")
    return node


def resolve_program(jwt: str, program_key: str) -> dict[str, Any]:
    root_id, locale, release = _resolve_root_node_id(jwt, program_key)
    node = fetch_program_tree(jwt, root_id)
    metadata = _metadata_from_release(release)
    if not metadata:
        # Construction path stores metadata on release; published path on component.
        construction = _construction_release(jwt, program_key, locale)
        if construction:
            metadata = _metadata_from_release(construction)
    node["_resolved_locale"] = locale
    node["_kind"] = node.get("semantic_type") or "Unknown"
    node["_metadata"] = metadata
    node["_unreleased"] = release.get("_unreleased", False)
    return node


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split())


def _fetch_vtt_text(url: str) -> str:
    try:
        resp = requests.get(url, timeout=30)
        if not resp.ok:
            return ""
        text = _clean(resp.text)
        if len(text) > VTT_MAX_CHARS:
            return text[:VTT_MAX_CHARS] + "..."
        return text
    except Exception:
        return ""


def _fetch_masterfiles_download_url(
    jwt: str,
    workspace_id: str,
    branch_id: int,
) -> str | None:
    """Resolve a signed GCS download URL for workspace starter/master files."""
    url = f"{WORKSPACE_PROVISIONER_BASE}/masterfiles/download/{workspace_id}"
    try:
        resp = requests.get(
            url,
            headers=_auth_headers(jwt),
            params={"branch_id": branch_id},
            timeout=_TIMEOUT,
        )
    except Exception:
        return None
    if resp.status_code == 404:
        return None
    if resp.status_code in (401, 403):
        raise UdacityAPIError(
            f"workspace-provisioner HTTP {resp.status_code}: JWT cannot download masterfiles."
        )
    if not resp.ok:
        return None
    try:
        body = resp.json()
    except Exception:
        return None
    download_url = body.get("download_url")
    return download_url if isinstance(download_url, str) and download_url else None


def _download_url_bytes(url: str) -> bytes:
    try:
        resp = requests.get(url, timeout=120)
        if resp.ok:
            return resp.content
    except Exception:
        pass
    return b""


def _extract_text_from_ipynb(data: bytes) -> str:
    try:
        nb = json.loads(data.decode("utf-8", errors="replace"))
    except Exception:
        return ""
    parts: list[str] = []
    for cell in nb.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        source = cell.get("source") or ""
        if isinstance(source, list):
            source = "".join(str(s) for s in source)
        source = str(source).strip()
        if source:
            parts.append(source)
    return "\n".join(parts)


def _archive_member_text(path: str, data: bytes) -> str:
    suffix = PurePosixPath(path).suffix.lower()
    if suffix == ".ipynb":
        return _extract_text_from_ipynb(data)
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _should_skip_archive_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    for part in parts:
        if part in SKIP_ARCHIVE_DIRS:
            return True
    suffix = PurePosixPath(path).suffix.lower()
    return suffix not in TEXT_EXTENSIONS


def _extract_text_from_archive(data: bytes, *, max_chars: int = WORKSPACE_FILES_MAX_CHARS) -> str:
    """Extract readable text from a masterfiles tar.gz archive."""
    if not data:
        return ""
    chunks: list[str] = []
    total = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                path = member.name or ""
                if _should_skip_archive_path(path):
                    continue
                if member.size > WORKSPACE_FILE_MAX_BYTES:
                    continue
                try:
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    raw = extracted.read()
                except Exception:
                    continue
                text = _clean(_archive_member_text(path, raw))
                if not text:
                    continue
                block = f"[path: {path}] {text}"
                if total + len(block) > max_chars:
                    remaining = max_chars - total
                    if remaining > 50:
                        chunks.append(block[:remaining] + "...")
                    break
                chunks.append(block)
                total += len(block)
    except Exception:
        return ""
    return "\n".join(chunks)


def _workspace_files_for_atom(
    atom: dict[str, Any],
    jwt: str,
    cache: dict[tuple[int, str, str], str],
) -> str:
    workspace_id = atom.get("workspace_id")
    branch_id = atom.get("branch_id")
    if not workspace_id or not branch_id:
        return ""
    try:
        branch_int = int(branch_id)
    except (TypeError, ValueError):
        return ""

    master_key = str(atom.get("master_archive_id") or "")
    cache_key = (branch_int, str(workspace_id), master_key)
    if cache_key in cache:
        return cache[cache_key]

    download_url = _fetch_masterfiles_download_url(jwt, str(workspace_id), branch_int)
    if not download_url:
        cache[cache_key] = ""
        return ""

    archive_bytes = _download_url_bytes(download_url)
    if not archive_bytes:
        cache[cache_key] = ""
        return ""

    text = _extract_text_from_archive(archive_bytes)
    cache[cache_key] = text
    return text


def _is_workspace_atom(atom: dict[str, Any]) -> bool:
    typ = (atom.get("__typename") or "").lower()
    semantic = (atom.get("semantic_type") or "").lower()
    return typ == "workspaceatom" or semantic in ("workspaceatom", "workspace")


def _atom_text(atom: dict[str, Any], *, workspace_files_text: str = "") -> str:
    semantic = atom.get("semantic_type") or atom.get("__typename") or ""
    title = atom.get("title") or ""
    parts: list[str] = []
    if title:
        parts.append(title)

    if atom.get("text"):
        parts.append(_clean(atom["text"]))

    if _is_workspace_atom(atom):
        if atom.get("name"):
            parts.append(f"name: {atom['name']}")
        if atom.get("instructor_notes"):
            parts.append(_clean(atom["instructor_notes"]))
        if atom.get("workspace_id"):
            parts.append(f"workspace_id: {atom['workspace_id']}")
        if atom.get("pool_id"):
            parts.append(f"pool_id: {atom['pool_id']}")
        if atom.get("main_default_path"):
            parts.append(f"main_default_path: {atom['main_default_path']}")
        config = atom.get("configuration")
        if config:
            try:
                parts.append(f"configuration: {json.dumps(config, ensure_ascii=False)[:2000]}")
            except (TypeError, ValueError):
                parts.append(f"configuration: {str(config)[:2000]}")
        if workspace_files_text:
            parts.append(f"[WorkspaceAtom files] {workspace_files_text}")

    question = atom.get("question")
    if isinstance(question, dict):
        prompt = _clean(question.get("prompt"))
        if prompt:
            parts.append(prompt)
        cp = question.get("complex_prompt")
        if isinstance(cp, dict) and cp.get("text"):
            parts.append(_clean(cp["text"]))
        cf = question.get("correct_feedback")
        if cf:
            parts.append(_clean(cf))
        answers = question.get("answers")
        if isinstance(answers, list):
            ans_texts = [
                _clean(a.get("text")) for a in answers if isinstance(a, dict) and a.get("text")
            ]
            if ans_texts:
                parts.append("Answers: " + " | ".join(ans_texts))
        concepts = question.get("concepts")
        if isinstance(concepts, list):
            con_texts = [
                _clean(c.get("text")) for c in concepts if isinstance(c, dict) and c.get("text")
            ]
            if con_texts:
                parts.append("Concepts: " + " | ".join(con_texts))

    video = atom.get("video")
    if isinstance(video, dict) and video.get("vtt_url"):
        vtt_url = video["vtt_url"]
        vtt_body = _fetch_vtt_text(vtt_url)
        if vtt_body:
            parts.append(f"Video transcript: {vtt_body}")
        else:
            parts.append(f"(video transcript url: {vtt_url})")

    if not parts:
        return ""
    prefix = f"[{semantic}] " if semantic else ""
    return prefix + " | ".join(parts)


def _concept_has_workspace(concept: dict[str, Any]) -> bool:
    for atom in concept.get("atoms") or []:
        if _is_workspace_atom(atom):
            return True
    return False


def _build_concept_context(
    concept: dict[str, Any],
    jwt: str,
    masterfiles_cache: dict[tuple[int, str, str], str],
) -> tuple[str, bool, int]:
    lines: list[str] = []
    files_included = False
    files_chars = 0
    for atom in concept.get("atoms") or []:
        workspace_files = ""
        if _is_workspace_atom(atom) and jwt:
            workspace_files = _workspace_files_for_atom(atom, jwt, masterfiles_cache)
            if workspace_files:
                files_included = True
                files_chars += len(workspace_files)
        text = _atom_text(atom, workspace_files_text=workspace_files)
        if text:
            lines.append(text)
    return "\n".join(lines), files_included, files_chars


def _walk_lessons_for_workspace(
    lessons: list[dict[str, Any]],
    jwt: str,
    masterfiles_cache: dict[tuple[int, str, str], str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for lesson in lessons or []:
        lesson_key = lesson.get("key") or ""
        lesson_title = lesson.get("title") or ""
        for concept in lesson.get("concepts") or []:
            if not _concept_has_workspace(concept):
                continue
            context, files_included, files_chars = _build_concept_context(
                concept, jwt, masterfiles_cache
            )
            results.append(
                {
                    "concept_key": concept.get("key") or "",
                    "concept_title": concept.get("title") or "",
                    "lesson_key": lesson_key,
                    "lesson_title": lesson_title,
                    "context_text": context,
                    "workspace_files_included": files_included,
                    "workspace_files_chars": files_chars,
                }
            )
    return results


def extract_workspace_concepts(
    program: dict[str, Any],
    jwt: str,
) -> list[dict[str, Any]]:
    """Find all concepts containing a WorkspaceAtom and extract atom + starter file context."""
    masterfiles_cache: dict[tuple[int, str, str], str] = {}
    concepts: list[dict[str, Any]] = []
    nd_parts = [p for p in (program.get("parts") or []) if p and p.get("key")]
    part_modules = program.get("modules") or []

    if nd_parts:
        for part in nd_parts:
            for module in part.get("modules") or []:
                concepts.extend(
                    _walk_lessons_for_workspace(
                        module.get("lessons"), jwt, masterfiles_cache
                    )
                )
    elif part_modules:
        for module in part_modules:
            concepts.extend(
                _walk_lessons_for_workspace(module.get("lessons"), jwt, masterfiles_cache)
            )
    elif program.get("concepts"):
        concepts.extend(
            _walk_lessons_for_workspace([program], jwt, masterfiles_cache)
        )
    else:
        concepts.extend(
            _walk_lessons_for_workspace(program.get("lessons") or [], jwt, masterfiles_cache)
        )
    return concepts


def _canonical_skill(name: str, allowed: list[str]) -> str | None:
    if name in allowed:
        return name
    lower_map = {s.lower(): s for s in allowed}
    return lower_map.get(name.lower())


def validate_skills(
    skills: Any,
    allowed: list[str],
    *,
    min_count: int = 1,
    max_count: int = 3,
) -> tuple[list[str], str | None]:
    """Return (canonical_skills, error_message). error_message is None on success."""
    if not isinstance(skills, list):
        return [], "recommended_skills is not a list"
    canonical: list[str] = []
    seen: set[str] = set()
    for raw in skills:
        if not isinstance(raw, str) or not raw.strip():
            continue
        match = _canonical_skill(raw.strip(), allowed)
        if not match:
            return [], f"skill not in program allowlist: {raw!r}"
        if match not in seen:
            seen.add(match)
            canonical.append(match)
    if len(canonical) < min_count:
        return [], f"expected {min_count}-{max_count} skills, got {len(canonical)}"
    if len(canonical) > max_count:
        canonical = canonical[:max_count]
    return canonical, None


def _build_user_prompt(
    program_key: str,
    program_title: str,
    concept: dict[str, Any],
    allowed_skills: list[str],
) -> str:
    skill_lines = "\n".join(f"- {s}" for s in allowed_skills)
    return (
        f"PROGRAM: {program_key} — {program_title}\n"
        f"ALLOWED SKILLS (pick 1–3 exact names from this list only):\n{skill_lines}\n\n"
        f"CONCEPT: {concept.get('concept_title', '')} (key: {concept.get('concept_key', '')})\n"
        "This concept includes a workspace activity. Tag skills the learner will practice, "
        "apply, or be meaningfully exposed to by engaging with the workspace and related "
        "content—not general program themes unrelated to this concept.\n\n"
        f"CONTENT:\n{concept.get('context_text', '') or '(no extractable content)'}"
    )


def _normalize_rationale(text: str) -> str:
    t = (text or "").strip()
    if len(t) <= RATIONALE_MAX_LENGTH:
        return t
    return t[:RATIONALE_MAX_LENGTH].rstrip()


SYSTEM_PROMPT = """You are a Udacity curriculum skills tagger for concepts that include a \
hands-on workspace activity.

Pick 1–3 skills from the ALLOWED SKILLS list that a learner will practice, apply, or be \
meaningfully exposed to if they complete the concept—especially by working through the \
workspace starter files, instructions, and supporting atoms (text, video, quizzes).

Ask: "What skill(s) will this learner have demonstrated—or at least engaged with—after \
finishing this concept?"

Rules:
- Copy skill names EXACTLY from the allowed list (same spelling and casing).
- Return 1 to 3 skills only.
- Tie choices to what the learner does or tries in the workspace (write code, debug, \
configure, analyze data, etc.), not tangential topics only mentioned in passing.
- Prefer demonstrated practice over passive exposure when both fit; if content is thin, \
pick the best-matching exposure-level skills from the list.
- rationale: max 255 characters; one short sentence; always call the analyzed item \
"concept" — never "lesson" or "course".
"""

STRICT_SYSTEM_PROMPT = SYSTEM_PROMPT + "\nIMPORTANT: You previously returned invalid skill names. \
Only use exact strings from the allowed list."


def _openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def recommend_skills(
    api_key: str,
    user_prompt: str,
    allowed_skills: list[str],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    strict: bool = False,
) -> tuple[SkillRecommendation | None, list[str], str | None]:
    """Call OpenAI once. Returns (parsed, validated_skills, validation_error)."""
    client = _openai_client(api_key)
    system = STRICT_SYSTEM_PROMPT if strict else SYSTEM_PROMPT
    try:
        resp = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            response_format=SkillRecommendation,
            temperature=temperature,
        )
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            return None, [], "LLM returned no parsed response"
        validated, err = validate_skills(parsed.recommended_skills, allowed_skills)
        if err:
            return parsed, [], err
        return parsed, validated, None
    except Exception as e:
        return None, [], f"OpenAI error: {e}"


def run_with_consensus(
    api_key: str,
    user_prompt: str,
    allowed_skills: list[str],
    *,
    n_runs: int = 3,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Run N LLM calls and compute majority-vote consensus."""
    run_skills: list[list[str]] = []
    run_rationales: list[str] = []
    validation_errors: list[str] = []

    for i in range(n_runs):
        parsed, skills, err = recommend_skills(
            api_key,
            user_prompt,
            allowed_skills,
            model=model,
            temperature=temperature,
        )
        if err and parsed:
            # Retry once with strict prompt on validation failure.
            parsed, skills, err = recommend_skills(
                api_key,
                user_prompt,
                allowed_skills,
                model=model,
                temperature=temperature,
                strict=True,
            )
        if err:
            validation_errors.append(f"run {i + 1}: {err}")
            run_skills.append([])
            run_rationales.append(
                _normalize_rationale(parsed.rationale) if parsed else ""
            )
        else:
            run_skills.append(skills)
            run_rationales.append(
                _normalize_rationale(parsed.rationale) if parsed else ""
            )

    threshold = math.ceil(n_runs / 2)
    vote_counter: Counter[str] = Counter()
    for skills in run_skills:
        for s in skills:
            vote_counter[s] += 1

    consensus = [s for s, count in vote_counter.items() if count >= threshold]
    # Preserve allowlist order for stable output.
    consensus_ordered = [s for s in allowed_skills if s in consensus]

    low_confidence = [
        s for s, count in vote_counter.items() if count < threshold and count > 0
    ]

    agreement_scores: list[float] = []
    for skills in run_skills:
        if not skills:
            agreement_scores.append(0.0)
            continue
        overlap = len(set(skills) & set(consensus_ordered))
        agreement_scores.append(overlap / max(len(skills), 1))
    agreement_score = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.0

    skill_votes = {s: vote_counter[s] for s in vote_counter}

    rationale = ""
    for r, skills in zip(run_rationales, run_skills):
        if set(skills) == set(consensus_ordered) and r:
            rationale = r
            break
    if not rationale and run_rationales:
        rationale = run_rationales[0]

    status = "ok"
    if not consensus_ordered and validation_errors:
        status = "error"
    elif validation_errors:
        status = "partial"

    return {
        "consensus_skills": consensus_ordered,
        "run_skills": run_skills,
        "skill_votes": skill_votes,
        "low_confidence_skills": low_confidence,
        "agreement_score": round(agreement_score, 3),
        "rationale": rationale,
        "validation_status": status,
        "validation_errors": validation_errors,
    }


def tag_concept(
    api_key: str,
    program_key: str,
    program_title: str,
    concept: dict[str, Any],
    allowed_skills: list[str],
    *,
    consensus_enabled: bool = True,
    n_runs: int = 3,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    user_prompt = _build_user_prompt(program_key, program_title, concept, allowed_skills)
    if consensus_enabled and n_runs > 1:
        outcome = run_with_consensus(
            api_key,
            user_prompt,
            allowed_skills,
            n_runs=n_runs,
            model=model,
        )
        return {
            "concept_key": concept.get("concept_key", ""),
            "concept_title": concept.get("concept_title", ""),
            "lesson_key": concept.get("lesson_key", ""),
            "lesson_title": concept.get("lesson_title", ""),
            "context_text": concept.get("context_text", ""),
            "llm_input_text": user_prompt,
            "workspace_files_included": concept.get("workspace_files_included", False),
            "workspace_files_chars": concept.get("workspace_files_chars", 0),
            "consensus_skills": outcome["consensus_skills"],
            "run_skills": outcome["run_skills"],
            "skill_votes": outcome["skill_votes"],
            "low_confidence_skills": outcome["low_confidence_skills"],
            "agreement_score": outcome["agreement_score"],
            "rationale": outcome["rationale"],
            "validation_status": outcome["validation_status"],
            "validation_errors": outcome["validation_errors"],
        }

    # Single run
    parsed, skills, err = recommend_skills(api_key, user_prompt, allowed_skills, model=model)
    if err and parsed:
        parsed, skills, err = recommend_skills(
            api_key, user_prompt, allowed_skills, model=model, strict=True
        )
    status = "ok" if not err else "error"
    return {
        "concept_key": concept.get("concept_key", ""),
        "concept_title": concept.get("concept_title", ""),
        "lesson_key": concept.get("lesson_key", ""),
        "lesson_title": concept.get("lesson_title", ""),
        "context_text": concept.get("context_text", ""),
        "llm_input_text": user_prompt,
        "workspace_files_included": concept.get("workspace_files_included", False),
        "workspace_files_chars": concept.get("workspace_files_chars", 0),
        "consensus_skills": skills,
        "run_skills": [skills] if skills else [],
        "skill_votes": {s: 1 for s in skills},
        "low_confidence_skills": [],
        "agreement_score": 1.0 if skills else 0.0,
        "rationale": _normalize_rationale(parsed.rationale) if parsed else "",
        "validation_status": status,
        "validation_errors": [err] if err else [],
    }


def analyze_program(
    program_key: str,
    jwt: str,
    openai_api_key: str,
    *,
    include_prerequisites: bool = False,
    consensus_enabled: bool = True,
    n_runs: int = 3,
    model: str = DEFAULT_MODEL,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """
    Full pipeline: resolve program, extract workspace concepts, tag each with LLM.

    progress_callback: optional callable(message, current=0, total=0) for UI updates.
      total > 0 means concept tagging progress (current/total).
    """
    def _progress(msg: str, *, current: int = 0, total: int = 0) -> None:
        if progress_callback:
            progress_callback(msg, current=current, total=total)

    _progress("Resolving program key...")
    program = resolve_program(jwt, program_key)
    metadata = program.get("_metadata")
    teaches_skills = _skills_from_metadata(metadata, include_prerequisites=False)
    allowed_skills = _skills_from_metadata(
        metadata, include_prerequisites=include_prerequisites
    )
    if not allowed_skills:
        raise UdacityAPIError(
            "No teaches_skills found in program metadata. "
            "The program may lack skill metadata or the JWT cannot read it."
        )

    program_title = program.get("title") or program_key
    _progress("Extracting workspace concepts and starter files...")
    workspace_concepts = extract_workspace_concepts(program, jwt)

    if not workspace_concepts:
        return {
            "program_meta": {
                "key": program_key,
                "title": program_title,
                "teaches_skills": teaches_skills,
                "allowed_skills": allowed_skills,
                "locale": program.get("_resolved_locale"),
                "unreleased": program.get("_unreleased", False),
                "workspace_concept_count": 0,
            },
            "workspace_concepts": [],
            "results": [],
        }

    results: list[dict[str, Any]] = []
    total = len(workspace_concepts)
    for i, concept in enumerate(workspace_concepts, 1):
        _progress(
            f"Tagging: {concept.get('concept_title', '')}",
            current=i,
            total=total,
        )
        results.append(
            tag_concept(
                openai_api_key,
                program_key,
                program_title,
                concept,
                allowed_skills,
                consensus_enabled=consensus_enabled,
                n_runs=n_runs,
                model=model,
            )
        )

    return {
        "program_meta": {
            "key": program_key,
            "title": program_title,
            "teaches_skills": teaches_skills,
            "allowed_skills": allowed_skills,
            "locale": program.get("_resolved_locale"),
            "unreleased": program.get("_unreleased", False),
            "workspace_concept_count": len(workspace_concepts),
        },
        "workspace_concepts": workspace_concepts,
        "results": results,
    }
