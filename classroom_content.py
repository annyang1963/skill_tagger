"""Udacity classroom-content client, workspace concept extraction, and skill tagging via OpenAI."""
from __future__ import annotations

import io
import json
import math
import re
import tarfile
import urllib.parse
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
# Query params of a workspace launch URL that name the file/folder opened for the learner.
DEFAULT_PATH_QUERY_KEYS = ("folder", "file", "path")
# Query params holding a JSON launch config rather than a path.
DEFAULT_PATH_CONFIG_KEYS = ("ulab", "blueprint")

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
      id
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
      dependencies {
        parent_node_id
        current {
          root_node_id
          component {
            key
            type
            title
          }
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
    branch_id
  }
  component(key: $key, locale: $locale) {
    metadata {
      difficulty_level { name uri }
      teaches_skills { name uri }
      prerequisite_skills { name uri }
    }
    branches(type: "CONSTRUCTION") {
      id
      root_node_id
      dependencies {
        parent_node_id
        current {
          root_node_id
          component {
            key
            type
            title
          }
        }
      }
    }
  }
}

fragment conceptFields on Concept {
  id
  key
  title
  is_public
  progress_key
  branch_id
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
  semantic_type
  branch_id
  concepts { ...conceptFields }
}

fragment moduleFields on Module {
  id
  key
  title
  semantic_type
  branch_id
  lessons { ...lessonFields }
}

fragment partFields on Part {
  id
  key
  title
  summary
  is_optional
  is_public
  semantic_type
  branch_id
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
    branch_id
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
    component = data.get("component") or {}
    branches = component.get("branches") or []
    construction_branch = branches[0] if branches else {}
    return {
        "id": construction_branch.get("id") or node.get("branch_id"),
        "root_node_id": node.get("id"),
        "root_node": {"id": node.get("id"), "title": node.get("title")},
        "component": {"metadata": component.get("metadata")},
        "dependencies": construction_branch.get("dependencies") or [],
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
    node["_branch_id"] = release.get("id") or node.get("branch_id")
    node["_child_components"] = _child_components_from_release(release)
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


def _default_path_target(main_default_path: str | None) -> str | None:
    """Resolve the folder/file a concept opens for the learner, or None if it opens the root.

    main_default_path is a workspace launch URL. Modern code-server workspaces use
    `/?folder=%2Fworkspace%2F<dir>`, legacy Jupyter ones a bare `/notebooks/<file>`, and
    legacy ulab/blueprint blobs a JSON config whose defaultPath is the workspace root.
    """
    raw = (main_default_path or "").strip()
    if not raw:
        return None

    parsed = urllib.parse.urlparse(raw)
    query = urllib.parse.parse_qs(parsed.query)
    candidate = ""

    for key in DEFAULT_PATH_QUERY_KEYS:
        values = query.get(key) or []
        if values and values[0].strip():
            candidate = values[0]
            break

    if not candidate:
        for key in DEFAULT_PATH_CONFIG_KEYS:
            values = query.get(key) or []
            if not values:
                continue
            try:
                blob = json.loads(urllib.parse.unquote(values[0]))
            except (TypeError, ValueError):
                return None
            if isinstance(blob, dict):
                candidate = str(blob.get("defaultPath") or "")
            break

    if not candidate and not query:
        candidate = parsed.path

    candidate = urllib.parse.unquote(candidate or "")
    parts = [p for p in candidate.split("/") if p and p not in (".", "..")]
    return "/".join(parts) or None


def _normalize_archive_name(name: str) -> str:
    """Archive members are stored as './a/b'; compare them as 'a/b'."""
    normalized = (name or "").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")


def _resolve_archive_scope(names: set[str], target: str) -> list[str]:
    """Map a default path onto the archive paths it refers to.

    The default path is rooted at the container (`/workspace/...`, `/notebooks/...`)
    while the archive is rooted at the repo, so leading segments are dropped one at a
    time until a segment suffix matches. Returns [] when nothing matches.
    """
    parts = [p for p in target.split("/") if p]
    for start in range(len(parts)):
        candidate = "/".join(parts[start:])
        roots: set[str] = set()
        for name in names:
            if name == candidate or name.startswith(candidate + "/"):
                roots.add(candidate)
                continue
            marker = "/" + candidate
            if name.endswith(marker):
                roots.add(name)
                continue
            index = name.find(marker + "/")
            if index != -1:
                roots.add(name[: index + len(marker)])
        if roots:
            return sorted(roots)
    return []


def _is_within_scope(name: str, roots: list[str]) -> bool:
    return any(name == root or name.startswith(root + "/") for root in roots)


def _is_readme(name: str) -> bool:
    return PurePosixPath(name).name.lower().startswith("readme")


def _ancestor_doc_paths(names: set[str], roots: list[str]) -> set[str]:
    """READMEs in folders above the exercise folder — they carry the task description."""
    ancestors: set[str] = set()
    for root in roots:
        parts = root.split("/")
        for depth in range(len(parts)):
            ancestors.add("/".join(parts[:depth]))
    docs: set[str] = set()
    for name in names:
        if not _is_readme(name) or _is_within_scope(name, roots):
            continue
        parent = name.rsplit("/", 1)[0] if "/" in name else ""
        if parent in ancestors:
            docs.add(name)
    return docs


def _extract_text_from_archive(
    data: bytes,
    *,
    scope_target: str | None = None,
    max_chars: int = WORKSPACE_FILES_MAX_CHARS,
) -> tuple[str, dict[str, Any]]:
    """Extract readable text from a masterfiles tar.gz archive.

    A workspace often holds one folder per exercise while a concept practices only one
    of them, so when scope_target resolves inside the archive only that folder (plus
    ancestor READMEs) is read. Falls back to the whole archive when it does not resolve.
    """
    info: dict[str, Any] = {
        "scope_target": scope_target or "",
        "scope_paths": [],
        "scoped": False,
    }
    if not data:
        return "", info

    chunks: list[str] = []
    total = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            names = {_normalize_archive_name(m.name) for m in members}

            roots = _resolve_archive_scope(names, scope_target) if scope_target else []
            docs = _ancestor_doc_paths(names, roots) if roots else set()
            info["scope_paths"] = roots
            info["scoped"] = bool(roots)

            selected = []
            for member in members:
                name = _normalize_archive_name(member.name)
                if roots and not (_is_within_scope(name, roots) or name in docs):
                    continue
                selected.append((name, member))
            # In-scope files first so truncation never drops the exercise for a README.
            selected.sort(key=lambda item: (item[0] in docs, item[0]))

            for name, member in selected:
                if _should_skip_archive_path(name):
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
                text = _clean(_archive_member_text(name, raw))
                if not text:
                    continue
                block = f"[path: {name}] {text}"
                if total + len(block) > max_chars:
                    remaining = max_chars - total
                    if remaining > 50:
                        chunks.append(block[:remaining] + "...")
                    break
                chunks.append(block)
                total += len(block)
    except Exception:
        return "", info
    return "\n".join(chunks), info


def _workspace_files_for_atom(
    atom: dict[str, Any],
    jwt: str,
    cache: dict[tuple[int, str, str], bytes],
) -> tuple[str, dict[str, Any]]:
    """Return (text, scope info) for this atom's slice of the workspace masterfiles.

    The archive bytes are cached, not the extracted text: concepts sharing a workspace
    each scope to a different exercise folder, so one download serves all of them.
    """
    empty_info: dict[str, Any] = {"scope_target": "", "scope_paths": [], "scoped": False}
    workspace_id = atom.get("workspace_id")
    branch_id = atom.get("branch_id")
    if not workspace_id or not branch_id:
        return "", empty_info
    try:
        branch_int = int(branch_id)
    except (TypeError, ValueError):
        return "", empty_info

    master_key = str(atom.get("master_archive_id") or "")
    cache_key = (branch_int, str(workspace_id), master_key)
    if cache_key in cache:
        archive_bytes = cache[cache_key]
    else:
        download_url = _fetch_masterfiles_download_url(jwt, str(workspace_id), branch_int)
        archive_bytes = _download_url_bytes(download_url) if download_url else b""
        cache[cache_key] = archive_bytes

    if not archive_bytes:
        return "", empty_info

    scope_target = _default_path_target(atom.get("main_default_path"))
    return _extract_text_from_archive(archive_bytes, scope_target=scope_target)


def _is_workspace_atom(atom: dict[str, Any]) -> bool:
    typ = (atom.get("__typename") or "").lower()
    semantic = (atom.get("semantic_type") or "").lower()
    return typ == "workspaceatom" or semantic in ("workspaceatom", "workspace")


def _atom_text(
    atom: dict[str, Any],
    *,
    workspace_files_text: str = "",
    workspace_scope: dict[str, Any] | None = None,
) -> str:
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
            scope_paths = (workspace_scope or {}).get("scope_paths") or []
            if scope_paths:
                label = (
                    "[WorkspaceAtom files — exercise folder for this concept: "
                    f"{'; '.join(scope_paths)}]"
                )
            else:
                label = "[WorkspaceAtom files]"
            parts.append(f"{label} {workspace_files_text}")

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
    masterfiles_cache: dict[tuple[int, str, str], bytes],
) -> tuple[str, bool, int, dict[str, Any]]:
    lines: list[str] = []
    files_included = False
    files_chars = 0
    scope_paths: list[str] = []
    scope_targets: list[str] = []
    unresolved = False

    for atom in concept.get("atoms") or []:
        workspace_files = ""
        scope: dict[str, Any] = {}
        if _is_workspace_atom(atom) and jwt:
            workspace_files, scope = _workspace_files_for_atom(atom, jwt, masterfiles_cache)
            if workspace_files:
                files_included = True
                files_chars += len(workspace_files)
                target = scope.get("scope_target") or ""
                if target:
                    scope_targets.append(target)
                if scope.get("scoped"):
                    scope_paths.extend(scope.get("scope_paths") or [])
                elif target:
                    unresolved = True
        text = _atom_text(atom, workspace_files_text=workspace_files, workspace_scope=scope)
        if text:
            lines.append(text)

    if scope_paths:
        status = "scoped"
    elif unresolved:
        status = "unresolved"
    elif files_included:
        status = "full_workspace"
    else:
        status = ""

    scope_info = {
        "workspace_scope_status": status,
        "workspace_scope_paths": scope_paths,
        "workspace_default_paths": scope_targets,
    }
    return "\n".join(lines), files_included, files_chars, scope_info


def _child_components_from_release(release: dict[str, Any] | None) -> dict[int, dict[str, str]]:
    """Map each nested child-component root_node_id to {key, type, title}."""
    out: dict[int, dict[str, str]] = {}
    if not release:
        return out
    for dep in release.get("dependencies") or []:
        if not isinstance(dep, dict):
            continue
        current = dep.get("current") or {}
        root_id = current.get("root_node_id")
        if root_id is None:
            continue
        comp = current.get("component") or dep.get("component") or {}
        out[int(root_id)] = {
            "key": (comp.get("key") or "").strip(),
            "type": (comp.get("type") or "").strip(),
            "title": (comp.get("title") or "").strip(),
        }
    return out


def _enclosing_child_component(
    node: dict[str, Any] | None,
    enclosing: dict[str, str] | None,
    child_by_root_id: dict[int, dict[str, str]],
    parent_branch_id: int | None,
) -> dict[str, str] | None:
    """Return the child component this node belongs to, if any.

    Prefers an explicit GraphQL dependency whose root is this node (so a nested
    ls inside a cd is more specific than the enclosing cd). Falls back to a
    branch_id mismatch when dependency data is missing.
    """
    if not node:
        return enclosing
    nid = node.get("id")
    if nid is not None:
        try:
            matched = child_by_root_id.get(int(nid))
        except (TypeError, ValueError):
            matched = None
        if matched:
            return matched
    nkey = (node.get("key") or "").strip()
    if nkey:
        for child in child_by_root_id.values():
            if child.get("key") == nkey:
                return child
    if enclosing is not None:
        return enclosing
    bid = node.get("branch_id")
    if parent_branch_id is None or bid is None:
        return None
    try:
        if int(bid) == int(parent_branch_id):
            return None
    except (TypeError, ValueError):
        return None
    return {
        "key": (node.get("key") or "").strip(),
        "type": (node.get("semantic_type") or "").strip(),
        "title": (node.get("title") or "").strip(),
    }


def _skip_reason_for_child(child: dict[str, str]) -> str:
    key = child.get("key") or ""
    title = child.get("title") or ""
    if key and title:
        label = f"{key} ({title})"
    else:
        label = key or title or "unknown"
    return (
        f"Skill tagging skipped because this concept is in child component {label}."
    )


def _workspace_concept_record(
    concept: dict[str, Any],
    lesson: dict[str, Any],
    *,
    context: str = "",
    files_included: bool = False,
    files_chars: int = 0,
    child: dict[str, str] | None = None,
    scope_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    skipped = child is not None
    scope = scope_info or {}
    return {
        "concept_key": concept.get("key") or "",
        "concept_title": concept.get("title") or "",
        "lesson_key": lesson.get("key") or "",
        "lesson_title": lesson.get("title") or "",
        "context_text": context,
        "workspace_files_included": files_included,
        "workspace_files_chars": files_chars,
        "workspace_scope_status": scope.get("workspace_scope_status", ""),
        "workspace_scope_paths": scope.get("workspace_scope_paths") or [],
        "workspace_default_paths": scope.get("workspace_default_paths") or [],
        "skipped": skipped,
        "skip_reason": _skip_reason_for_child(child) if child else "",
        "child_component_key": (child or {}).get("key") or "",
        "child_component_type": (child or {}).get("type") or "",
        "child_component_title": (child or {}).get("title") or "",
    }


def _walk_lessons_for_workspace(
    lessons: list[dict[str, Any]],
    jwt: str,
    masterfiles_cache: dict[tuple[int, str, str], bytes],
    *,
    child_by_root_id: dict[int, dict[str, str]],
    parent_branch_id: int | None,
    enclosing_child: dict[str, str] | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for lesson in lessons or []:
        child = _enclosing_child_component(
            lesson, enclosing_child, child_by_root_id, parent_branch_id
        )
        for concept in lesson.get("concepts") or []:
            if not _concept_has_workspace(concept):
                continue
            concept_child = _enclosing_child_component(
                concept, child, child_by_root_id, parent_branch_id
            )
            if concept_child:
                results.append(
                    _workspace_concept_record(concept, lesson, child=concept_child)
                )
                continue
            context, files_included, files_chars, scope_info = _build_concept_context(
                concept, jwt, masterfiles_cache
            )
            results.append(
                _workspace_concept_record(
                    concept,
                    lesson,
                    context=context,
                    files_included=files_included,
                    files_chars=files_chars,
                    scope_info=scope_info,
                )
            )
    return results


def extract_workspace_concepts(
    program: dict[str, Any],
    jwt: str,
) -> list[dict[str, Any]]:
    """Find workspace concepts owned by this component; skip nested child components."""
    masterfiles_cache: dict[tuple[int, str, str], bytes] = {}
    concepts: list[dict[str, Any]] = []
    child_by_root_id = program.get("_child_components") or {}
    parent_branch_id = program.get("_branch_id")
    try:
        parent_branch_id = int(parent_branch_id) if parent_branch_id is not None else None
    except (TypeError, ValueError):
        parent_branch_id = None

    nd_parts = [p for p in (program.get("parts") or []) if p and p.get("key")]
    part_modules = program.get("modules") or []

    def walk(
        lessons: list[dict[str, Any]] | None,
        enclosing: dict[str, str] | None,
    ) -> None:
        concepts.extend(
            _walk_lessons_for_workspace(
                lessons or [],
                jwt,
                masterfiles_cache,
                child_by_root_id=child_by_root_id,
                parent_branch_id=parent_branch_id,
                enclosing_child=enclosing,
            )
        )

    if nd_parts:
        for part in nd_parts:
            part_child = _enclosing_child_component(
                part, None, child_by_root_id, parent_branch_id
            )
            for module in part.get("modules") or []:
                module_child = _enclosing_child_component(
                    module, part_child, child_by_root_id, parent_branch_id
                )
                walk(module.get("lessons"), module_child)
    elif part_modules:
        program_child = _enclosing_child_component(
            program, None, child_by_root_id, parent_branch_id
        )
        for module in part_modules:
            module_child = _enclosing_child_component(
                module, program_child, child_by_root_id, parent_branch_id
            )
            walk(module.get("lessons"), module_child)
    elif program.get("concepts"):
        # Analyzing a lesson component directly — its concepts are in-component.
        walk([program], None)
    else:
        program_child = _enclosing_child_component(
            program, None, child_by_root_id, parent_branch_id
        )
        walk(program.get("lessons") or [], program_child)
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
    scope_paths = concept.get("workspace_scope_paths") or []
    scope_note = ""
    if scope_paths:
        scope_note = (
            "The workspace files below are only the exercise folder this concept opens "
            f"({'; '.join(scope_paths)}), not the whole workspace.\n"
        )
    return (
        f"PROGRAM: {program_key} — {program_title}\n"
        f"ALLOWED SKILLS (pick 1–3 exact names from this list only):\n{skill_lines}\n\n"
        f"CONCEPT: {concept.get('concept_title', '')} (key: {concept.get('concept_key', '')})\n"
        "This concept includes a workspace activity. Tag skills the learner will practice, "
        "apply, or be meaningfully exposed to by engaging with the workspace and related "
        "content—not general program themes unrelated to this concept.\n"
        f"{scope_note}\n"
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
            "workspace_scope_status": concept.get("workspace_scope_status", ""),
            "workspace_scope_paths": concept.get("workspace_scope_paths") or [],
            "workspace_default_paths": concept.get("workspace_default_paths") or [],
            "consensus_skills": outcome["consensus_skills"],
            "run_skills": outcome["run_skills"],
            "skill_votes": outcome["skill_votes"],
            "low_confidence_skills": outcome["low_confidence_skills"],
            "agreement_score": outcome["agreement_score"],
            "rationale": outcome["rationale"],
            "validation_status": outcome["validation_status"],
            "validation_errors": outcome["validation_errors"],
            "skipped": False,
            "skip_reason": "",
            "child_component_key": "",
            "child_component_type": "",
            "child_component_title": "",
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
        "workspace_scope_status": concept.get("workspace_scope_status", ""),
        "workspace_scope_paths": concept.get("workspace_scope_paths") or [],
        "workspace_default_paths": concept.get("workspace_default_paths") or [],
        "consensus_skills": skills,
        "run_skills": [skills] if skills else [],
        "skill_votes": {s: 1 for s in skills},
        "low_confidence_skills": [],
        "agreement_score": 1.0 if skills else 0.0,
        "rationale": _normalize_rationale(parsed.rationale) if parsed else "",
        "validation_status": status,
        "validation_errors": [err] if err else [],
        "skipped": False,
        "skip_reason": "",
        "child_component_key": "",
        "child_component_type": "",
        "child_component_title": "",
    }


def _skipped_tag_result(concept: dict[str, Any]) -> dict[str, Any]:
    return {
        "concept_key": concept.get("concept_key", ""),
        "concept_title": concept.get("concept_title", ""),
        "lesson_key": concept.get("lesson_key", ""),
        "lesson_title": concept.get("lesson_title", ""),
        "context_text": "",
        "llm_input_text": "",
        "workspace_files_included": False,
        "workspace_files_chars": 0,
        "workspace_scope_status": "",
        "workspace_scope_paths": [],
        "workspace_default_paths": [],
        "consensus_skills": [],
        "run_skills": [],
        "skill_votes": {},
        "low_confidence_skills": [],
        "agreement_score": 0.0,
        "rationale": "",
        "validation_status": "skipped",
        "validation_errors": [],
        "skipped": True,
        "skip_reason": concept.get("skip_reason") or _skip_reason_for_child(
            {
                "key": concept.get("child_component_key") or "",
                "title": concept.get("child_component_title") or "",
            }
        ),
        "child_component_key": concept.get("child_component_key", ""),
        "child_component_type": concept.get("child_component_type", ""),
        "child_component_title": concept.get("child_component_title", ""),
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
    skipped_count = sum(1 for c in workspace_concepts if c.get("skipped"))
    tagged_count = len(workspace_concepts) - skipped_count

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
                "tagged_concept_count": 0,
                "skipped_child_concept_count": 0,
            },
            "workspace_concepts": [],
            "results": [],
        }

    results: list[dict[str, Any]] = []
    total = len(workspace_concepts)
    for i, concept in enumerate(workspace_concepts, 1):
        if concept.get("skipped"):
            child_key = concept.get("child_component_key") or "child component"
            _progress(
                f"Skipping (in {child_key}): {concept.get('concept_title', '')}",
                current=i,
                total=total,
            )
            results.append(_skipped_tag_result(concept))
            continue
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
            "tagged_concept_count": tagged_count,
            "skipped_child_concept_count": skipped_count,
        },
        "workspace_concepts": workspace_concepts,
        "results": results,
    }
