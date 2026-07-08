"""Export DST-SNN PyTorch checkpoints into snn-chat-lab compatible JSON."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc


HTML_TAG_TOKENS = {
    "a", "abbr", "address", "article", "aside", "audio", "b", "body", "br", "button", "canvas",
    "caption", "class", "code", "data", "details", "dialog", "div", "doctype", "em", "embed",
    "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6",
    "head", "header", "hr", "html", "html5", "iframe", "img", "input", "label", "legend", "li",
    "link", "main", "meta", "nav", "noscript", "ol", "option", "path", "picture", "pre", "role",
    "script", "section", "select", "slot", "source", "span", "style", "submit", "svg", "table",
    "tbody", "td", "template", "textarea", "tfoot", "th", "thead", "title", "tr", "track", "ul",
    "video", "viewbox", "xml",
}
URL_FRAGMENT_TOKENS = {
    "http", "https", "www", "com", "net", "org", "jp", "co", "io", "cdn", "static", "assets",
    "asset", "gstatic", "googleusercontent", "cloudfront", "akamaized", "blob", "mp4", "webm",
    "m3u8", "jpeg", "jpg", "png", "gif", "svg",
}


def _token_weight(token: str) -> float:
    if token.startswith(("image:", "audio:", "video:", "text:", "body:")):
        return 0.8
    if ":" in token:
        return 0.68
    return 1.0


def _modality(token: str) -> str:
    return token.split(":", 1)[0] if ":" in token else "text"


def _chat_token(raw_token: str) -> str:
    parts = [part for part in str(raw_token).lower().split(":") if part]
    if not parts:
        return ""
    modality = parts[0]
    leaf = parts[-1]
    if modality == "text":
        return leaf
    if modality == "body" and len(parts) >= 4 and parts[2] == "target":
        return leaf
    if modality in {"audio", "video"} and "label" in parts:
        return leaf
    if modality == "image" and "alt" in parts:
        return leaf
    return ":".join(parts)


def _is_noisy_chat_token(token: str) -> bool:
    value = str(token or "").lower().strip()
    leaf = value.split(":")[-1]
    if not value or leaf in HTML_TAG_TOKENS or leaf in URL_FRAGMENT_TOKENS:
        return True
    if leaf.isdigit():
        return not (len(leaf) == 4 and 1900 <= int(leaf) <= 2100)
    if re.fullmatch(r"[a-f0-9]{8,}", leaf):
        return True
    if re.fullmatch(r"[a-z0-9]{12,}", leaf) and not re.search(r"[aeiouぁ-んァ-ン一-龥]", leaf):
        return True
    if re.fullmatch(r"[a-z]{1,2}[0-9]+", leaf) or re.fullmatch(r"[0-9]+[a-z]{1,2}", leaf):
        return True
    return False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # Older PyTorch.
        return torch.load(checkpoint_path, map_location="cpu")


def checkpoint_to_chat_payload(checkpoint_path: Path) -> dict[str, Any]:
    state = _load_checkpoint(checkpoint_path)
    feature_space = state.get("feature_space", {})
    token_to_index = dict(feature_space.get("token_to_index", {}))
    index_to_tokens = {
        int(index): list(tokens)
        for index, tokens in feature_space.get("index_to_tokens", {}).items()
    }
    relations = state.get("relations", {}).get("relations", {})
    steps = int(state.get("steps", 0))

    tokens = sorted(token_to_index, key=lambda token: (token_to_index[token], token))
    token_to_id: dict[str, int] = {}
    display_to_id: dict[str, int] = {}
    vocabulary: list[dict[str, Any]] = []
    for token in tokens:
        display_token = _chat_token(token)
        if _is_noisy_chat_token(display_token):
            continue
        modality = _modality(token)
        if display_token in display_to_id:
            token_id = display_to_id[display_token]
            token_to_id[token] = token_id
            neuron = vocabulary[token_id]
            neuron["count"] = float(neuron.get("count", 0)) + 1
            if modality not in neuron["modalities"]:
                neuron["modalities"].append(modality)
            neuron.setdefault("rawTokens", []).append(token)
            continue
        token_id = len(vocabulary)
        display_to_id[display_token] = token_id
        token_to_id[token] = token_id
        vocabulary.append({
            "id": token_id,
            "token": display_token,
            "rawTokens": [token],
            "count": 1 + steps * 0.02,
            "v": 0,
            "threshold": 1,
            "positiveSpikeMass": 1,
            "negativeSpikeMass": 0,
            "stability": 0.12,
            "importance": _token_weight(token),
            "role": "semantic",
            "primaryModality": modality,
            "modalities": [modality],
            "sourceIndex": token_to_index[token],
        })

    associations: list[dict[str, Any]] = []
    seen_edges: set[tuple[int, int]] = set()

    def add_edge(pre_token: str, post_token: str, weight: float, stability: float = 0.1, relation_kind: str = "association") -> None:
        if pre_token not in token_to_id or post_token not in token_to_id:
            return
        pre = token_to_id[pre_token]
        post = token_to_id[post_token]
        if pre == post:
            return
        key = (pre, post)
        if key in seen_edges:
            return
        seen_edges.add(key)
        associations.append({
            "id": len(associations),
            "pre": pre,
            "post": post,
            "w": max(-1.0, min(1.0, weight)),
            "aPre": 0,
            "aPost": 0,
            "stability": max(0.0, min(1.0, stability)),
            "replayCount": 1,
            "lastUpdatedStep": steps,
            "d1Go": max(0.0, weight),
            "d2NoGo": max(0.0, -weight),
            "rewardPrediction": 0,
            "relationKind": relation_kind,
        })

    for relation in relations.values():
        rel_tokens = relation.get("tokens", [])
        if len(rel_tokens) != 2:
            continue
        w = _safe_float(relation.get("w"), 0.04)
        stability = _safe_float(relation.get("stability"), 0.1)
        add_edge(rel_tokens[0], rel_tokens[1], w, stability, "cross_modal")
        add_edge(rel_tokens[1], rel_tokens[0], w * 0.92, stability, "cross_modal")

    weight = state.get("model", {}).get("dendrite.weight")
    if weight is not None:
        weight = weight.detach().cpu()
        max_inputs = min(weight.shape[0], len(index_to_tokens))
        for input_index in range(max_inputs):
            source_tokens = index_to_tokens.get(input_index, [])[:3]
            if not source_tokens:
                continue
            values, output_indices = torch.topk(weight[input_index].abs(), k=min(4, weight.shape[1]))
            for value, output_index in zip(values.tolist(), output_indices.tolist()):
                target_tokens = index_to_tokens.get(int(output_index), [])[:3]
                if not target_tokens:
                    continue
                signed = _safe_float(weight[input_index, output_index].item())
                association_weight = max(-0.75, min(0.75, signed))
                if abs(association_weight) < 0.025:
                    association_weight = 0.025 if signed >= 0 else -0.025
                for source in source_tokens:
                    for target in target_tokens:
                        add_edge(source, target, association_weight, min(0.7, abs(value)), "dst_weight")

    filtered_relations = [
        relation for relation in relations.values()
        if all(token in token_to_id for token in relation.get("tokens", []))
    ][:1200]
    observations = [{
        "step": steps,
        "eventType": "dst_checkpoint",
        "url": str(checkpoint_path),
        "title": checkpoint_path.name,
        "tokenCount": len(vocabulary),
        "reward": 0,
        "salience": 1,
    }]
    model = {
        "version": 1,
        "domain": "dst-web",
        "savedAt": int(checkpoint_path.stat().st_mtime * 1000) if checkpoint_path.exists() else 0,
        "source": "dst-snn-pytorch",
        "stats": {
            "steps": steps,
            "observations": steps,
            "totalTokens": len(vocabulary),
            "positiveSpikes": len(vocabulary),
            "associations": len(associations),
            "crossModalRelations": len(filtered_relations),
        },
        "vocabulary": vocabulary,
        "associations": associations,
        "observations": observations,
        "crossModalRelations": filtered_relations,
    }
    return {
        "kind": "dst-snn-chat-model",
        "version": 1,
        "sourceCheckpoint": str(checkpoint_path),
        "exportedAt": checkpoint_path.stat().st_mtime if checkpoint_path.exists() else None,
        "model": model,
    }


def export_checkpoint_for_chat(checkpoint_path: Path, output_path: Path | None = None) -> Path:
    output_path = output_path or checkpoint_path.with_suffix(".chat.json")
    payload = checkpoint_to_chat_payload(checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
