"""Playwright-driven online learner for DST-SNN.

The learner observes web pages through modular sensors, converts observations
into sparse spike trains, trains the PyTorch DST-SNN sequentially, and maintains
cross-modal relation weights between text, image, audio, video, and body/action
features.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import html
import json
import math
from pathlib import Path
import random
import re
import time
from typing import Any, Iterable, Optional, Protocol

try:
    import torch
    from torch import Tensor
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install Pillow with `pip install -r requirements-dst-snn.txt`.") from exc

from .dendritic_layer import DendriticSNN
from .chat_export import export_checkpoint_for_chat


TOKEN_RE = re.compile(r"[\w\-ぁ-んァ-ン一-龥]{2,}", re.UNICODE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
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
SENSITIVE_LABEL_RE = re.compile(
    r"password|pass|secret|token|card|payment|checkout|delete|logout|sign out|"
    r"remove|unsubscribe|purchase|buy|order|submit payment|"
    r"パスワード|秘密|カード|支払い|購入|削除|退会|ログアウト|注文|精算",
    re.IGNORECASE,
)


def clean_text_for_learning(text: str) -> str:
    value = html.unescape(str(text or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    value = URL_RE.sub(" ", value)
    value = value.replace("_", " ").replace("/", " ").replace("\\", " ")
    return re.sub(r"\s+", " ", value).strip()


def is_learning_token(token: str) -> bool:
    value = normalize_token(token)
    if not value or value in HTML_TAG_TOKENS or value in URL_FRAGMENT_TOKENS:
        return False
    if value.isdigit():
        return len(value) == 4 and 1900 <= int(value) <= 2100
    if len(value) <= 1:
        return False
    if re.fullmatch(r"[a-f0-9]{8,}", value):
        return False
    if re.fullmatch(r"[a-z0-9]{12,}", value) and not re.search(r"[aeiouぁ-んァ-ン一-龥]", value):
        return False
    if re.fullmatch(r"[a-z]{1,2}[0-9]+", value) or re.fullmatch(r"[0-9]+[a-z]{1,2}", value):
        return False
    return True


@dataclass
class ModuleObservation:
    module: str
    modality: str
    tokens: list[str]
    salience: float = 0.5
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BodyAction:
    kind: str
    target: str = ""
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WebObservation:
    url: str
    title: str
    modules: list[ModuleObservation]
    action: Optional[BodyAction] = None
    at: float = field(default_factory=time.time)


def action_record(action: Optional[BodyAction]) -> dict[str, Any]:
    if not action:
        return {}
    return {
        "kind": action.kind,
        "target": action.target,
        "success": action.success,
        "metadata": action.metadata,
    }


class ObservationModule(Protocol):
    name: str
    modality: str

    async def observe(self, page: Any) -> list[ModuleObservation]:
        ...


def normalize_token(value: str) -> str:
    return value.strip().lower()


def tokenize(text: str, limit: int = 160) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for match in TOKEN_RE.finditer(clean_text_for_learning(text)):
        token = normalize_token(match.group(0))
        if not is_learning_token(token):
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= limit:
            break
    return tokens


def stable_hash(value: str) -> int:
    return int(hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest(), 16)


class FeatureSpace:
    """Fixed-dimensional sparse feature space for growing modules.

    Tokens are namespaced by module/modality and hashed into a fixed input space
    so new modalities can be added without resizing the DST-SNN layer.
    """

    def __init__(self, in_features: int) -> None:
        self.in_features = in_features
        self.token_to_index: dict[str, int] = {}
        self.index_to_tokens: dict[int, list[str]] = defaultdict(list)

    def index(self, token: str) -> int:
        if token not in self.token_to_index:
            idx = stable_hash(token) % self.in_features
            self.token_to_index[token] = idx
            if token not in self.index_to_tokens[idx]:
                self.index_to_tokens[idx].append(token)
        return self.token_to_index[token]

    def encode(self, observations: Iterable[ModuleObservation], time_steps: int) -> tuple[Tensor, Tensor, list[dict[str, Any]]]:
        spikes = torch.zeros(time_steps, self.in_features)
        target = torch.zeros(self.in_features)
        active: list[dict[str, Any]] = []
        for obs_index, obs in enumerate(observations):
            base_time = int((obs_index + 1) * time_steps / 7) % time_steps
            for token_index, token in enumerate(obs.tokens):
                namespaced = f"{obs.modality}:{obs.module}:{token}"
                idx = self.index(namespaced)
                t = (base_time + token_index * 3 + stable_hash(namespaced) % max(1, time_steps // 3)) % time_steps
                intensity = max(0.05, min(1.0, obs.salience))
                spikes[t, idx] = max(spikes[t, idx], intensity)
                target[idx] = 1.0
                active.append({
                    "token": namespaced,
                    "index": idx,
                    "module": obs.module,
                    "modality": obs.modality,
                    "salience": intensity,
                })
        return spikes, target, active

    def state_dict(self) -> dict[str, Any]:
        return {
            "in_features": self.in_features,
            "token_to_index": self.token_to_index,
            "index_to_tokens": {str(k): v for k, v in self.index_to_tokens.items()},
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "FeatureSpace":
        obj = cls(int(state["in_features"]))
        obj.token_to_index = dict(state.get("token_to_index", {}))
        obj.index_to_tokens = defaultdict(list, {int(k): list(v) for k, v in state.get("index_to_tokens", {}).items()})
        return obj


class CrossModalRelationMemory:
    """Symmetric relation weights between active feature tokens."""

    def __init__(self, max_relations: int = 12000) -> None:
        self.max_relations = max_relations
        self.relations: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _key(left: str, right: str) -> str:
        a, b = sorted([left, right])
        return f"{a}\t{b}"

    def update(self, active: list[dict[str, Any]], reward: float, salience: float) -> int:
        by_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in active:
            by_modality[item["modality"]].append(item)
        modalities = sorted(by_modality)
        updates = 0
        for i, left_modality in enumerate(modalities):
            for right_modality in modalities[i + 1:]:
                left_items = sorted(by_modality[left_modality], key=lambda item: item["salience"], reverse=True)[:12]
                right_items = sorted(by_modality[right_modality], key=lambda item: item["salience"], reverse=True)[:12]
                for left in left_items:
                    for right in right_items:
                        key = self._key(left["token"], right["token"])
                        rel = self.relations.setdefault(key, {
                            "tokens": sorted([left["token"], right["token"]]),
                            "modalities": sorted([left_modality, right_modality]),
                            "w": 0.04,
                            "coactivity": 0.0,
                            "stability": 0.0,
                            "updates": 0,
                        })
                        coactivity = math.sqrt(max(0.01, left["salience"]) * max(0.01, right["salience"]))
                        rel["coactivity"] = rel["coactivity"] * 0.9 + coactivity * 0.1
                        rel["w"] = max(-1.0, min(1.0, rel["w"] + (0.015 + salience * 0.02) * coactivity * reward))
                        rel["stability"] = max(0.0, min(1.0, rel["stability"] + abs(rel["w"]) * 0.0008 + coactivity * 0.0005))
                        rel["updates"] += 1
                        updates += 1
        if len(self.relations) > self.max_relations:
            keep = sorted(
                self.relations.items(),
                key=lambda item: abs(item[1]["w"]) + item[1]["stability"] + item[1]["coactivity"] * 0.08,
                reverse=True,
            )[:self.max_relations]
            self.relations = dict(keep)
        return updates

    def top(self, limit: int = 32) -> list[dict[str, Any]]:
        return [
            value for _, value in sorted(
                self.relations.items(),
                key=lambda item: abs(item[1]["w"]) + item[1]["stability"] + item[1]["coactivity"] * 0.08,
                reverse=True,
            )[:limit]
        ]

    def state_dict(self) -> dict[str, Any]:
        return {"max_relations": self.max_relations, "relations": self.relations}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "CrossModalRelationMemory":
        obj = cls(int(state.get("max_relations", 12000)))
        obj.relations = dict(state.get("relations", {}))
        return obj


class TextModule:
    name = "dom-text"
    modality = "text"

    async def observe(self, page: Any) -> list[ModuleObservation]:
        data = await page.evaluate(
            """() => {
                const blocked = "script,style,noscript,template,svg,canvas,code,pre,kbd,samp";
                const visible = (node) => {
                    const rect = node.getBoundingClientRect?.();
                    const style = getComputedStyle(node);
                    return rect && rect.width > 0 && rect.height > 0
                        && style.visibility !== "hidden" && style.display !== "none";
                };
                const cleanNodeText = (node) => {
                    const clone = node.cloneNode(true);
                    clone.querySelectorAll(blocked).forEach((child) => child.remove());
                    return (clone.innerText || clone.textContent || "").replace(/\\s+/g, " ").trim();
                };
                const textSelectors = [
                    "main p", "main li", "main blockquote",
                    "article p", "article li", "article blockquote",
                    "[role=main] p", "[role=main] li",
                    "section p", "section li"
                ].join(",");
                const headingSelectors = "h1,h2,h3,h4";
                const bodyClone = document.body?.cloneNode(true);
                bodyClone?.querySelectorAll(blocked).forEach((child) => child.remove());
                const semanticText = Array.from(document.querySelectorAll(textSelectors))
                    .filter(visible)
                    .map(cleanNodeText)
                    .filter((text) => text.length >= 2)
                    .join(" ");
                const fallbackText = (bodyClone?.innerText || bodyClone?.textContent || "").replace(/\\s+/g, " ");
                const headings = Array.from(document.querySelectorAll(headingSelectors))
                    .filter(visible)
                    .map(cleanNodeText)
                    .join(" ");
                return {
                    title: document.title || "",
                    url: location.href,
                    text: (semanticText || fallbackText || "").slice(0, 9000),
                    headings
                };
            }"""
        )
        tokens = [
            *[f"title:{token}" for token in tokenize(data.get("title", ""), 24)],
            *[f"heading:{token}" for token in tokenize(data.get("headings", ""), 48)],
            *tokenize(data.get("text", ""), 180),
        ]
        return [ModuleObservation(self.name, self.modality, tokens, 0.74, data.get("url", ""))]


class VisualModule:
    name = "page-visual"
    modality = "image"

    async def observe(self, page: Any) -> list[ModuleObservation]:
        observations: list[ModuleObservation] = []
        try:
            screenshot = await page.screenshot(full_page=False, type="png")
            tokens = image_feature_tokens(screenshot, prefix="screenshot")
            observations.append(ModuleObservation(self.name, self.modality, tokens, 0.72, "viewport"))
        except Exception:
            observations.append(ModuleObservation(self.name, self.modality, ["pixels:unreadable"], 0.25, "viewport"))

        image_meta = await page.evaluate(
            """() => Array.from(document.querySelectorAll("img")).slice(0, 16).map(img => {
                const rect = img.getBoundingClientRect();
                return {
                    alt: img.alt || img.getAttribute("aria-label") || img.title || "",
                    width: Math.round(rect.width),
                    height: Math.round(rect.height),
                    visible: rect.width > 24 && rect.height > 24 && rect.bottom >= 0 && rect.right >= 0
                        && rect.top <= innerHeight && rect.left <= innerWidth
                };
            }).filter(item => item.visible)"""
        )
        meta_tokens: list[str] = []
        for item in image_meta:
            area = int(item.get("width", 0)) * int(item.get("height", 0))
            meta_tokens.append("image:size:large" if area > 180000 else "image:size:small")
            meta_tokens.extend(f"alt:{token}" for token in tokenize(item.get("alt", ""), 16))
        if meta_tokens:
            observations.append(ModuleObservation("image-dom", "image", meta_tokens[:160], 0.62, "img"))
        return observations


class MediaModule:
    name = "dom-media"
    modality = "video"

    async def observe(self, page: Any) -> list[ModuleObservation]:
        media_items = await page.evaluate(
            """() => Array.from(document.querySelectorAll("video,audio")).slice(0, 12).map((el) => ({
                kind: el.tagName.toLowerCase(),
                duration: Number.isFinite(el.duration) ? el.duration : 0,
                currentTime: Number.isFinite(el.currentTime) ? el.currentTime : 0,
                paused: el.paused,
                muted: el.muted,
                volume: Number.isFinite(el.volume) ? el.volume : 0,
                width: el.videoWidth || el.clientWidth || 0,
                height: el.videoHeight || el.clientHeight || 0,
                text: (
                    el.getAttribute("aria-label") || el.title ||
                    Array.from(el.querySelectorAll("track[label]")).map(track => track.label).join(" ")
                ).slice(0, 400)
            }))"""
        )
        observations: list[ModuleObservation] = []
        for item in media_items:
            kind = item.get("kind", "media")
            modality = "audio" if kind == "audio" else "video"
            progress = 0
            if item.get("duration"):
                progress = int(10 * float(item.get("currentTime", 0)) / max(1.0, float(item["duration"])))
            tokens = [
                f"{modality}:present",
                f"{modality}:{'paused' if item.get('paused') else 'playing'}",
                f"{modality}:{'muted' if item.get('muted') else 'audible'}",
                f"{modality}:progress:{progress}",
                f"{modality}:volume:{round(float(item.get('volume', 0)) * 10)}",
            ]
            if modality == "video":
                tokens.append("video:size:large" if int(item.get("width", 0)) >= 960 else "video:size:small")
            tokens.extend(f"label:{token}" for token in tokenize(item.get("text", ""), 24))
            observations.append(ModuleObservation("media-state", modality, tokens, 0.68, kind, item))
        return observations


class BodyModule:
    name = "browser-body"
    modality = "body"

    async def observe(self, page: Any) -> list[ModuleObservation]:
        viewport = await page.evaluate("""() => ({ x: scrollX, y: scrollY, w: innerWidth, h: innerHeight })""")
        tokens = [
            "body:browser",
            f"body:scroll:{min(9, int(int(viewport.get('y', 0)) / 800))}",
            "body:viewport:tall" if int(viewport.get("h", 0)) > 850 else "body:viewport:compact",
        ]
        return [ModuleObservation(self.name, self.modality, tokens, 0.42, "viewport", viewport)]


def image_feature_tokens(png_bytes: bytes, prefix: str = "image") -> list[str]:
    from io import BytesIO

    image = Image.open(BytesIO(png_bytes)).convert("RGB").resize((16, 16))
    pixels = list(image.getdata())
    brightness = 0.0
    saturation = 0.0
    edge = 0.0
    r_sum = g_sum = b_sum = 0.0
    previous = None
    for r, g, b in pixels:
        mx, mn = max(r, g, b), min(r, g, b)
        luma = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        brightness += luma
        saturation += 0.0 if mx == 0 else (mx - mn) / mx
        r_sum += r
        g_sum += g
        b_sum += b
        if previous is not None:
            edge += abs(luma - previous)
        previous = luma
    count = max(1, len(pixels))
    brightness /= count
    saturation /= count
    edge /= count
    hue = dominant_hue(r_sum / count, g_sum / count, b_sum / count)
    return [
        f"{prefix}:brightness:{bucket(brightness)}",
        f"{prefix}:saturation:{bucket(saturation)}",
        f"{prefix}:edge:{bucket(min(1.0, edge * 6))}",
        f"{prefix}:hue:{hue}",
    ]


def bucket(value: float) -> str:
    if value < 0.28:
        return "low"
    if value > 0.68:
        return "high"
    return "mid"


def dominant_hue(r: float, g: float, b: float) -> str:
    if max(r, g, b) - min(r, g, b) < 18:
        return "neutral"
    if r >= g and r >= b:
        return "warm" if g > b else "red"
    if g >= r and g >= b:
        return "yellow-green" if r > b else "green"
    return "purple-blue" if r > g else "blue"


class DstWebLearner:
    """Online DST-SNN learner with pluggable web observation modules."""

    def __init__(
        self,
        in_features: int = 512,
        time_steps: int = 48,
        branches: int = 16,
        max_delay: int = 16,
        lr: float = 2e-3,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.time_steps = time_steps
        self.feature_space = FeatureSpace(in_features)
        self.model = DendriticSNN(
            in_features=in_features,
            out_features=in_features,
            num_branches=branches,
            max_delay=max_delay,
            learnable_delay=True,
            threshold=0.85,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.relations = CrossModalRelationMemory()
        self.modules: list[ObservationModule] = [TextModule(), VisualModule(), MediaModule(), BodyModule()]
        self.steps = 0

    def add_module(self, module: ObservationModule) -> None:
        self.modules.append(module)

    async def observe(self, page: Any, action: Optional[BodyAction] = None) -> WebObservation:
        module_observations: list[ModuleObservation] = []
        for module in self.modules:
            try:
                module_observations.extend(await module.observe(page))
            except Exception as exc:
                module_observations.append(ModuleObservation(
                    module=getattr(module, "name", module.__class__.__name__),
                    modality=getattr(module, "modality", "unknown"),
                    tokens=[f"module-error:{type(exc).__name__.lower()}"],
                    salience=0.1,
                    source="error",
                ))
        if action:
            action_tokens = [
                f"action:{action.kind}",
                f"success:{int(action.success)}",
            ]
            action_tokens.extend(f"target:{token}" for token in tokenize(action.target, 16))
            if action.metadata.get("urlChanged"):
                action_tokens.append("change:url")
            if action.metadata.get("titleChanged"):
                action_tokens.append("change:title")
            if action.metadata.get("textChanged"):
                action_tokens.append("change:text")
            text_delta = int(action.metadata.get("textDelta") or 0)
            if text_delta:
                action_tokens.append("change:text:grow" if text_delta > 0 else "change:text:shrink")
            if action.metadata.get("errorText"):
                action_tokens.extend(["change:error", *tokenize(str(action.metadata["errorText"]), 24)])
            module_observations.append(ModuleObservation(
                "body-action",
                "body",
                action_tokens,
                0.7,
                "action",
                action.metadata,
            ))
        title = await page.title()
        return WebObservation(page.url, title, module_observations, action)

    def train_observation(self, observation: WebObservation) -> dict[str, Any]:
        spikes, target, active = self.feature_space.encode(observation.modules, self.time_steps)
        x = spikes.unsqueeze(0).to(self.device)
        y = target.unsqueeze(0).to(self.device)
        out = self.model(x)
        logits = out["membrane"].amax(dim=1)
        spike_rate = out["spikes"].mean()
        loss = F.binary_cross_entropy_with_logits(logits, y) + 0.0008 * spike_rate
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        salience = max([obs.salience for obs in observation.modules], default=0.1)
        novelty = sum(1 for item in active if self.feature_space.token_to_index.get(item["token"]) is not None) / max(1, len(active))
        reward = max(0.05, min(1.0, 0.2 + salience * 0.35 + novelty * 0.1))
        relation_updates = self.relations.update(active, reward=reward, salience=salience)
        self.steps += 1
        return {
            "step": self.steps,
            "url": observation.url,
            "title": observation.title,
            "loss": float(loss.detach().cpu()),
            "spike_rate": float(spike_rate.detach().cpu()),
            "active_features": len(active),
            "relation_updates": relation_updates,
            "relations": len(self.relations.relations),
            "last_action": action_record(observation.action),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "feature_space": self.feature_space.state_dict(),
            "relations": self.relations.state_dict(),
            "steps": self.steps,
        }, path)
        path.with_suffix(".relations.json").write_text(
            json.dumps({"top": self.relations.top(80)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        export_checkpoint_for_chat(path)

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.feature_space = FeatureSpace.from_state_dict(state["feature_space"])
        self.relations = CrossModalRelationMemory.from_state_dict(state["relations"])
        self.steps = int(state.get("steps", 0))


class PlaywrightWebTrainer:
    def __init__(self, learner: DstWebLearner, headless: bool = True) -> None:
        self.learner = learner
        self.headless = headless
        self.rng = random.Random()
        self.action_step = 0
        self.visited_urls: set[str] = set()
        self.navigation_count = 0
        self.max_navigations = 24

    async def run(
        self,
        start_urls: list[str],
        steps_per_url: int,
        allow_clicks: bool,
        allow_inputs: bool,
        allow_navigation: bool,
        checkpoint: Optional[Path],
        endless: bool = False,
        max_total_steps: int = 0,
        save_every: int = 10,
        step_delay_ms: int = 900,
        max_navigations: int = 24,
    ) -> None:
        self.max_navigations = max(0, max_navigations)
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install Playwright and browsers: `pip install -r requirements-dst-snn.txt && playwright install chromium`.") from exc

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            page = await browser.new_page(viewport={"width": 1280, "height": 900})
            try:
                total_steps = 0
                url_index = 0
                while True:
                    url = start_urls[url_index % len(start_urls)]
                    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    await page.wait_for_timeout(max(0, step_delay_ms))
                    action: Optional[BodyAction] = BodyAction("page_view", url)
                    local_steps = 0
                    while endless or local_steps < steps_per_url:
                        if max_total_steps > 0 and total_steps >= max_total_steps:
                            if checkpoint:
                                self.learner.save(checkpoint)
                            return
                        observation = await self.learner.observe(page, action)
                        metrics = self.learner.train_observation(observation)
                        print(json.dumps(metrics, ensure_ascii=False))
                        total_steps += 1
                        local_steps += 1
                        if checkpoint and save_every > 0 and self.learner.steps % save_every == 0:
                            self.learner.save(checkpoint)
                        if max_total_steps > 0 and total_steps >= max_total_steps:
                            if checkpoint:
                                self.learner.save(checkpoint)
                            return
                        if not endless and local_steps >= steps_per_url:
                            break
                        if endless and not allow_navigation and len(start_urls) > 1 and local_steps >= steps_per_url:
                            break
                        action = await self.act(page, allow_clicks, allow_inputs, allow_navigation)
                        print(json.dumps({
                            "event": "action",
                            "step": self.learner.steps,
                            "url": page.url,
                            "action": action_record(action),
                        }, ensure_ascii=False))
                        await page.wait_for_timeout(max(0, step_delay_ms))
                    if not endless:
                        url_index += 1
                        if url_index >= len(start_urls):
                            break
                    else:
                        url_index += 1 if not allow_navigation and len(start_urls) > 1 else 0
                if checkpoint:
                    self.learner.save(checkpoint)
            finally:
                if checkpoint:
                    self.learner.save(checkpoint)
                await browser.close()

    async def act(self, page: Any, allow_clicks: bool, allow_inputs: bool, allow_navigation: bool) -> BodyAction:
        self.action_step += 1
        candidates = await self.action_candidates(page)
        links = [item for item in candidates["links"] if item.get("href") not in self.visited_urls]
        buttons = candidates["buttons"]
        inputs = candidates["inputs"]

        should_navigate = (
            allow_navigation
            and links
            and self.navigation_count < self.max_navigations
            and (self.action_step % 4 == 0 or self.rng.random() < 0.34)
        )
        should_input = allow_inputs and inputs and (self.action_step % 5 == 0 or self.rng.random() < 0.22)
        should_click = allow_clicks and buttons and (self.action_step % 3 == 0 or self.rng.random() < 0.28)

        ordered_actions = []
        if should_navigate:
            ordered_actions.append("navigate")
        if should_input:
            ordered_actions.append("input")
        if should_click:
            ordered_actions.append("button")
        ordered_actions.extend(["button", "input", "navigate", "scroll"])

        for action_kind in ordered_actions:
            if action_kind == "navigate" and allow_navigation and links and self.navigation_count < self.max_navigations:
                action = await self.try_navigate(page, links)
                if action:
                    return action
            if action_kind == "button" and allow_clicks and buttons:
                action = await self.try_button(page, buttons)
                if action:
                    return action
            if action_kind == "input" and allow_inputs and inputs:
                action = await self.try_input(page, inputs, allow_navigation)
                if action:
                    return action
            if action_kind == "scroll":
                return await self.scroll(page)
        return await self.scroll(page)

    async def action_candidates(self, page: Any) -> dict[str, list[dict[str, Any]]]:
        try:
            return await page.evaluate(
                """() => {
                    const unsafe = /(password|pass|secret|token|card|payment|checkout|delete|remove|unsubscribe|logout|log out|sign out|purchase|buy|order|submit payment|パスワード|秘密|カード|支払い|購入|削除|退会|ログアウト|注文|精算)/i;
                    const labelOf = (el) => (
                        el.innerText || el.value || el.getAttribute("aria-label") || el.getAttribute("placeholder") ||
                        el.title || el.name || el.id || ""
                    ).toString().replace(/\\s+/g, " ").trim().slice(0, 160);
                    const visible = (el) => {
                        const rect = el.getBoundingClientRect();
                        const style = getComputedStyle(el);
                        return style.visibility !== "hidden" && style.display !== "none" && Number(style.opacity) !== 0
                            && rect.width >= 16 && rect.height >= 8
                            && rect.bottom >= 0 && rect.top <= innerHeight && rect.right >= 0 && rect.left <= innerWidth;
                    };
                    const safeHref = (anchor) => {
                        const raw = anchor.getAttribute("href") || "";
                        if (!raw || raw.startsWith("#") || raw.startsWith("javascript:") || raw.startsWith("mailto:") || raw.startsWith("tel:")) return "";
                        if (anchor.hasAttribute("download")) return "";
                        const label = labelOf(anchor);
                        if (unsafe.test(label)) return "";
                        try {
                            const url = new URL(raw, location.href);
                            if (!/^https?:$/.test(url.protocol)) return "";
                            const here = `${location.origin}${location.pathname}${location.search}`;
                            const there = `${url.origin}${url.pathname}${url.search}`;
                            if (there === here) return "";
                            return url.href;
                        } catch {
                            return "";
                        }
                    };
                    const pageWords = Array.from(new Set(`${document.title} ${document.body?.innerText || ""}`
                        .toLowerCase()
                        .match(/[\\w\\-ぁ-んァ-ン一-龥]{3,}/g) || []))
                        .filter((word) => !unsafe.test(word))
                        .slice(0, 4);
                    const inputValue = (el) => {
                        const label = labelOf(el).toLowerCase();
                        if (/email|mail|phone|tel|password|pass|card|address|token|secret|メール|電話|住所|パスワード|秘密|カード/i.test(label)) return "";
                        if (/search|find|query|検索|探す/i.test(label)) return pageWords.length ? pageWords.slice(0, 3).join(" ") : "elfentier exploration";
                        return pageWords.length ? `explore ${pageWords.slice(0, 3).join(" ")}` : "elfentier exploration";
                    };
                    const links = Array.from(document.querySelectorAll("a[href]"))
                        .map((el, index) => ({ el, index, href: safeHref(el), label: labelOf(el) }))
                        .filter((item) => item.href && item.label && visible(item.el))
                        .slice(0, 40)
                        .map(({ index, href, label }) => ({ index, href, label }));
                    const buttonSelector = "button,[role=button],input[type=button],input[type=submit],input[type=reset],summary";
                    const buttons = Array.from(document.querySelectorAll(buttonSelector))
                        .map((el, index) => ({ el, index, label: labelOf(el), tagName: el.tagName || "" }))
                        .filter((item) => visible(item.el) && !item.el.disabled && item.el.getAttribute("aria-disabled") !== "true" && item.label && !unsafe.test(item.label))
                        .slice(0, 40)
                        .map(({ index, label, tagName }) => ({ index, label, tagName }));
                    const inputSelector = "input,textarea,[contenteditable=true],[contenteditable=plaintext-only]";
                    const allowedTypes = new Set(["text", "search", "url", "email", "tel", "number"]);
                    const inputs = Array.from(document.querySelectorAll(inputSelector))
                        .map((el, index) => {
                            const type = (el.getAttribute("type") || "text").toLowerCase();
                            return { el, index, type, label: labelOf(el), value: inputValue(el), tagName: el.tagName || "" };
                        })
                        .filter((item) => visible(item.el) && !item.el.disabled && !item.el.readOnly && item.el.getAttribute("aria-disabled") !== "true")
                        .filter((item) => item.el.tagName.toLowerCase() !== "input" || allowedTypes.has(item.type))
                        .filter((item) => item.value && !unsafe.test(item.label))
                        .slice(0, 24)
                        .map(({ index, type, label, value, tagName }) => ({ index, type, label, value, tagName }));
                    return { links, buttons, inputs };
                }"""
            )
        except Exception:
            return {"links": [], "buttons": [], "inputs": []}

    async def page_signature(self, page: Any) -> dict[str, Any]:
        try:
            data = await page.evaluate(
                """() => {
                    const clone = document.body?.cloneNode(true);
                    clone?.querySelectorAll("script,style,noscript,template,svg,canvas,code,pre,kbd,samp").forEach((node) => node.remove());
                    const text = (clone?.innerText || clone?.textContent || "").replace(/\\s+/g, " ").slice(0, 4000);
                    const errorText = Array.from(document.querySelectorAll('[role=alert], .error, .alert, [aria-live]'))
                        .map((node) => node.innerText || node.textContent || "")
                        .join(" ")
                        .replace(/\\s+/g, " ")
                        .slice(0, 600);
                    return { url: location.href, title: document.title || "", text, textLength: text.length, errorText };
                }"""
            )
        except Exception:
            return {"url": page.url, "title": "", "text": "", "textLength": 0, "errorText": ""}
        text = str(data.get("text", ""))
        data["textHash"] = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
        data.pop("text", None)
        return data

    async def finish_action(self, page: Any, before: dict[str, Any], kind: str, target: str, success: bool, metadata: dict[str, Any]) -> BodyAction:
        await page.wait_for_timeout(650)
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=2500)
        except Exception:
            pass
        after = await self.page_signature(page)
        text_delta = int(after.get("textLength", 0)) - int(before.get("textLength", 0))
        metadata.update({
            "beforeUrl": before.get("url", ""),
            "afterUrl": after.get("url", ""),
            "urlChanged": before.get("url") != after.get("url"),
            "titleChanged": before.get("title") != after.get("title"),
            "textChanged": before.get("textHash") != after.get("textHash"),
            "textDelta": text_delta,
            "errorText": after.get("errorText", ""),
        })
        return BodyAction(kind, target[:160], success, metadata)

    async def try_navigate(self, page: Any, links: list[dict[str, Any]]) -> Optional[BodyAction]:
        for item in self.rng.sample(links[: min(12, len(links))], k=min(4, len(links))):
            href = str(item.get("href") or "")
            label = str(item.get("label") or href)
            if not href or SENSITIVE_LABEL_RE.search(label):
                continue
            before = await self.page_signature(page)
            try:
                self.visited_urls.add(str(before.get("url") or ""))
                self.visited_urls.add(href)
                await page.goto(href, wait_until="domcontentloaded", timeout=30000)
                self.navigation_count += 1
                return await self.finish_action(page, before, "navigate_link", label, True, {"href": href})
            except Exception as exc:
                return await self.finish_action(page, before, "navigate_failed", label, False, {"href": href, "error": type(exc).__name__})
        return None

    async def try_button(self, page: Any, buttons: list[dict[str, Any]]) -> Optional[BodyAction]:
        selector = "button,[role=button],input[type=button],input[type=submit],input[type=reset],summary"
        for item in self.rng.sample(buttons[: min(12, len(buttons))], k=min(4, len(buttons))):
            label = str(item.get("label") or "button")
            if SENSITIVE_LABEL_RE.search(label):
                continue
            before = await self.page_signature(page)
            try:
                await page.locator(selector).nth(int(item["index"])).click(timeout=3500)
                return await self.finish_action(page, before, "click_button", label, True, {"tagName": item.get("tagName", "")})
            except Exception:
                continue
        return None

    async def try_input(self, page: Any, inputs: list[dict[str, Any]], allow_navigation: bool) -> Optional[BodyAction]:
        selector = "input,textarea,[contenteditable=true],[contenteditable=plaintext-only]"
        for item in self.rng.sample(inputs[: min(10, len(inputs))], k=min(4, len(inputs))):
            label = str(item.get("label") or "input")
            value = str(item.get("value") or "elfentier exploration")
            if SENSITIVE_LABEL_RE.search(label):
                continue
            before = await self.page_signature(page)
            try:
                handle = page.locator(selector).nth(int(item["index"]))
                await handle.fill(value, timeout=3500)
                if allow_navigation and re.search(r"search|find|query|検索|探す", label, re.IGNORECASE):
                    await handle.press("Enter", timeout=1500)
                return await self.finish_action(page, before, "input", label, True, {
                    "tagName": item.get("tagName", ""),
                    "inputType": item.get("type", ""),
                    "valuePreview": value[:80],
                })
            except Exception:
                continue
        return None

    async def scroll(self, page: Any) -> BodyAction:
        before = await self.page_signature(page)
        try:
            viewport = await page.evaluate("""() => ({ y: scrollY, maxY: Math.max(0, document.documentElement.scrollHeight - innerHeight), h: innerHeight })""")
            max_y = int(viewport.get("maxY", 0))
            current_y = int(viewport.get("y", 0))
            height = max(200, int(viewport.get("h", 900)))
            delta = -int(height * 0.62) if max_y > 0 and current_y / max(1, max_y) > 0.82 else int(height * (0.42 + self.rng.random() * 0.32))
        except Exception:
            delta = 520
        await page.mouse.wheel(0, delta)
        return await self.finish_action(page, before, "scroll", "page", True, {"deltaY": delta})


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", nargs="+")
    parser.add_argument("--steps-per-url", type=int, default=8)
    parser.add_argument("--in-features", type=int, default=512)
    parser.add_argument("--time-steps", type=int, default=48)
    parser.add_argument("--branches", type=int, default=16)
    parser.add_argument("--max-delay", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--headful", action="store_true")
    parser.add_argument("--allow-clicks", action="store_true")
    parser.add_argument("--allow-inputs", action="store_true")
    parser.add_argument("--allow-navigation", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/dst-web-learner.pt"))
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(endless=True)
    parser.add_argument("--endless", dest="endless", action="store_true", help="keep learning until interrupted")
    parser.add_argument("--once", dest="endless", action="store_false", help="run steps-per-url once and exit")
    parser.add_argument("--max-total-steps", type=int, default=0, help="optional cap for --endless; 0 means no cap")
    parser.add_argument("--save-every", type=int, default=10, help="checkpoint interval in learner steps")
    parser.add_argument("--step-delay-ms", type=int, default=900, help="delay between autonomous observations/actions")
    parser.add_argument("--max-navigations", type=int, default=24, help="maximum autonomous link navigations per run")
    return parser.parse_args(argv)


async def amain(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    learner = DstWebLearner(
        in_features=args.in_features,
        time_steps=args.time_steps,
        branches=args.branches,
        max_delay=args.max_delay,
        lr=args.lr,
        device=args.device,
    )
    if args.resume and args.checkpoint.exists():
        learner.load(args.checkpoint)
    trainer = PlaywrightWebTrainer(learner, headless=not args.headful)
    await trainer.run(
        args.urls,
        steps_per_url=args.steps_per_url,
        allow_clicks=args.allow_clicks,
        allow_inputs=args.allow_inputs,
        allow_navigation=args.allow_navigation,
        checkpoint=args.checkpoint,
        endless=args.endless,
        max_total_steps=args.max_total_steps,
        save_every=args.save_every,
        step_delay_ms=args.step_delay_ms,
        max_navigations=args.max_navigations,
    )


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
