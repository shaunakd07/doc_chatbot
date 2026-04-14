from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import httpx
from openai import OpenAI

from .. import config
from ..ingestion.doc_types import DOC_TYPE_ALIASES, infer_doc_type


LOGGER = logging.getLogger(__name__)


_ALIAS_NORMALIZER = re.compile(r"[^a-z0-9 ]+")


def _normalize_alias(value: str) -> str:
    collapsed = _ALIAS_NORMALIZER.sub(" ", str(value or "").strip().lower())
    return re.sub(r"\s+", " ", collapsed).strip()


def normalize_doc_type(value: str, *, use_alias_map: bool = True) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    if not raw:
        return ""
    normalized = _normalize_alias(raw.replace("_", " "))
    if not normalized:
        return ""
    if use_alias_map:
        if raw in set(DOC_TYPE_ALIASES.values()):
            return raw
        alias = DOC_TYPE_ALIASES.get(normalized)
        if alias:
            return alias
        compact = normalized.replace(" ", "")
        compact_alias = DOC_TYPE_ALIASES.get(compact)
        if compact_alias:
            return compact_alias
    return re.sub(r"_+", "_", normalized.replace(" ", "_")).strip("_")


@dataclass
class ClassificationResult:
    doc_type: str
    confidence: float
    scores: dict[str, float]
    provider: str
    model: str
    raw_response: dict[str, Any] | None = None


class DocumentClassifier(Protocol):
    def classify(
        self,
        *,
        file_path: Path | None,
        filename: str,
        auto_tags: Iterable[str],
        text_samples: Iterable[str],
    ) -> ClassificationResult:
        ...


class HeuristicDocumentClassifier:
    provider = "heuristic"
    model = "infer_doc_type_v1"

    def classify(
        self,
        *,
        file_path: Path | None,
        filename: str,
        auto_tags: Iterable[str],
        text_samples: Iterable[str],
    ) -> ClassificationResult:
        doc_type, confidence, scores = infer_doc_type(filename, auto_tags, text_samples)
        return ClassificationResult(
            doc_type=normalize_doc_type(doc_type) or "unknown",
            confidence=max(0.0, min(1.0, float(confidence or 0.0))),
            scores={str(key): float(value) for key, value in (scores or {}).items()},
            provider=self.provider,
            model=self.model,
        )


class SemanticOpenAIClassifier:
    provider = "semantic_openai"

    def __init__(
        self,
        *,
        model_id: str,
        api_key: str,
        timeout_sec: float,
    ) -> None:
        self.model = str(model_id or "gpt-4o-mini").strip() or "gpt-4o-mini"
        self.timeout_sec = max(5.0, float(timeout_sec))
        self._api_key = str(api_key or "").strip()
        self._client = OpenAI(api_key=self._api_key) if self._api_key else None

    def classify(
        self,
        *,
        file_path: Path | None,
        filename: str,
        auto_tags: Iterable[str],
        text_samples: Iterable[str],
    ) -> ClassificationResult:
        fallback = HeuristicDocumentClassifier().classify(
            file_path=file_path,
            filename=filename,
            auto_tags=auto_tags,
            text_samples=text_samples,
        )
        if self._client is None:
            LOGGER.warning("Semantic classifier selected but OPENAI_API_KEY is missing; using heuristic fallback")
            return fallback

        evidence_lines: list[str] = [f"filename: {str(filename or '').strip()}"]
        tags = [str(tag or "").strip() for tag in auto_tags if str(tag or "").strip()]
        if tags:
            evidence_lines.append(f"auto_tags: {', '.join(tags[:12])}")
        snippets: list[str] = []
        for sample in text_samples:
            value = str(sample or "").strip()
            if value:
                snippets.append(value[:420])
            if len(snippets) >= 12:
                break
        if snippets:
            evidence_lines.append("text_snippets:")
            for idx, snippet in enumerate(snippets, start=1):
                evidence_lines.append(f"{idx}. {snippet}")

        prompt = (
            "Classify this enterprise document into a short dynamic type label based on filename and content.\n"
            "Return strict JSON with keys:\n"
            '{\n'
            '  "label": "snake_case_label",\n'
            '  "confidence": 0.0_to_1.0,\n'
            '  "alternatives": [{"label":"...", "confidence":0.0_to_1.0}],\n'
            '  "reason": "short reason"\n'
            "}\n"
            "Guidelines:\n"
            "- Label must be semantic and specific if evidence supports it (e.g., technical_proposal, invoice, nda).\n"
            "- Do not force labels from a fixed taxonomy.\n"
            "- Keep confidence conservative.\n\n"
            + "\n".join(evidence_lines)
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an enterprise document classification assistant. Output JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=260,
                response_format={"type": "json_object"},
                timeout=self.timeout_sec,
            )
            content = str(response.choices[0].message.content or "").strip()
            parsed = json.loads(content) if content else {}
            if not isinstance(parsed, dict):
                raise ValueError("Semantic classifier returned non-object JSON")

            label = normalize_doc_type(
                str(
                    parsed.get("label")
                    or parsed.get("doc_type")
                    or parsed.get("predicted_type")
                    or ""
                ),
                use_alias_map=False,
            )
            confidence_raw = parsed.get("confidence")
            try:
                confidence = float(confidence_raw)
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            score_map: dict[str, float] = {}
            alternatives = parsed.get("alternatives")
            if isinstance(alternatives, list):
                for item in alternatives:
                    if not isinstance(item, dict):
                        continue
                    alt_label = normalize_doc_type(
                        str(item.get("label") or item.get("doc_type") or item.get("predicted_type") or ""),
                        use_alias_map=False,
                    )
                    if not alt_label:
                        continue
                    try:
                        alt_conf = float(item.get("confidence") or item.get("score") or 0.0)
                    except Exception:
                        alt_conf = 0.0
                    score_map[alt_label] = max(0.0, min(1.0, alt_conf))

            if label:
                score_map[label] = max(score_map.get(label, 0.0), confidence)

            if not label:
                return fallback

            if confidence <= 0 and score_map.get(label):
                confidence = float(score_map[label])

            return ClassificationResult(
                doc_type=label,
                confidence=max(0.0, min(1.0, confidence)),
                scores=score_map,
                provider=self.provider,
                model=self.model,
                raw_response=parsed,
            )
        except Exception as exc:
            LOGGER.warning("Semantic classifier failed for %s: %s", filename, exc)
            return fallback


class AzureDocumentIntelligenceClassifier:
    provider = "azure_document_intelligence"

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        classifier_id: str,
        api_version: str,
        timeout_sec: float,
        poll_interval_sec: float,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = str(api_key or "").strip()
        self.classifier_id = str(classifier_id or "").strip()
        self.api_version = str(api_version or "").strip()
        self.timeout_sec = max(5.0, float(timeout_sec))
        self.poll_interval_sec = max(0.2, float(poll_interval_sec))

    @property
    def model(self) -> str:
        return self.classifier_id or "custom_classifier"

    def classify(
        self,
        *,
        file_path: Path | None,
        filename: str,
        auto_tags: Iterable[str],
        text_samples: Iterable[str],
    ) -> ClassificationResult:
        if file_path is None or not file_path.exists() or not file_path.is_file():
            LOGGER.warning("Azure classifier fallback: missing file for %s", filename)
            return HeuristicDocumentClassifier().classify(
                file_path=file_path,
                filename=filename,
                auto_tags=auto_tags,
                text_samples=text_samples,
            )

        with file_path.open("rb") as handle:
            payload = handle.read()

        try:
            result_payload = self._run_analysis(payload)
            parsed = self._parse_result(result_payload)
            if parsed.doc_type == "unknown":
                return HeuristicDocumentClassifier().classify(
                    file_path=file_path,
                    filename=filename,
                    auto_tags=auto_tags,
                    text_samples=text_samples,
                )
            return parsed
        except Exception as exc:
            LOGGER.warning("Azure classifier failed for %s: %s", filename, exc)
            return HeuristicDocumentClassifier().classify(
                file_path=file_path,
                filename=filename,
                auto_tags=auto_tags,
                text_samples=text_samples,
            )

    def _run_analysis(self, content: bytes) -> dict[str, Any]:
        analyze_url = (
            f"{self.endpoint}/documentintelligence/documentClassifiers/{self.classifier_id}:analyze"
        )
        params = {"api-version": self.api_version}
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/octet-stream",
        }

        with httpx.Client(timeout=self.timeout_sec) as client:
            start = client.post(analyze_url, params=params, headers=headers, content=content)
            start.raise_for_status()
            operation_url = str(start.headers.get("Operation-Location") or start.headers.get("operation-location") or "").strip()
            if not operation_url:
                raise RuntimeError("Azure Document Intelligence did not return Operation-Location")

            deadline = time.monotonic() + self.timeout_sec
            while True:
                poll = client.get(
                    operation_url,
                    headers={"Ocp-Apim-Subscription-Key": self.api_key},
                )
                poll.raise_for_status()
                payload = poll.json()
                status = str(payload.get("status") or "").strip().lower()
                if status == "succeeded":
                    return payload
                if status == "failed":
                    raise RuntimeError(f"Azure Document Intelligence job failed: {payload}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Azure Document Intelligence classification timed out")
                time.sleep(self.poll_interval_sec)

    def _parse_result(self, payload: dict[str, Any]) -> ClassificationResult:
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        documents = result.get("documents") if isinstance(result.get("documents"), list) else []
        best_type = "unknown"
        best_conf = 0.0
        for item in documents:
            if not isinstance(item, dict):
                continue
            candidate = normalize_doc_type(
                str(item.get("docType") or item.get("doc_type") or ""),
                use_alias_map=False,
            )
            try:
                confidence = float(item.get("confidence") or 0.0)
            except Exception:
                confidence = 0.0
            if confidence >= best_conf:
                best_type = candidate or "unknown"
                best_conf = confidence

        score_map: dict[str, float] = {}
        classes = result.get("classes")
        if isinstance(classes, dict):
            for label, entry in classes.items():
                normalized = normalize_doc_type(str(label or ""), use_alias_map=False)
                if not normalized:
                    continue
                if isinstance(entry, dict):
                    try:
                        value = float(entry.get("confidence") or 0.0)
                    except Exception:
                        value = 0.0
                else:
                    try:
                        value = float(entry)
                    except Exception:
                        value = 0.0
                score_map[normalized] = value

        if best_type and best_type != "unknown":
            score_map.setdefault(best_type, float(best_conf))

        return ClassificationResult(
            doc_type=best_type or "unknown",
            confidence=max(0.0, min(1.0, float(best_conf))),
            scores=score_map,
            provider=self.provider,
            model=self.model,
            raw_response=payload,
        )


def build_document_classifier() -> DocumentClassifier:
    provider = str(config.DOC_TYPE_CLASSIFIER_PROVIDER or "heuristic").strip().lower()
    if provider in {"semantic", "semantic_openai", "dynamic", "openai_semantic"}:
        if config.OPENAI_API_KEY:
            return SemanticOpenAIClassifier(
                model_id=config.DOC_TYPE_SEMANTIC_MODEL,
                api_key=config.OPENAI_API_KEY,
                timeout_sec=config.DOC_TYPE_SEMANTIC_TIMEOUT_SEC,
            )
        LOGGER.warning("Semantic classifier configured but OPENAI_API_KEY is missing; using heuristic fallback")
    if provider in {"azure", "azure_document_intelligence", "azure_doc_intel"}:
        if config.AZURE_DOC_INTELLIGENCE_ENDPOINT and config.AZURE_DOC_INTELLIGENCE_API_KEY and config.AZURE_DOC_INTELLIGENCE_CLASSIFIER_ID:
            return AzureDocumentIntelligenceClassifier(
                endpoint=config.AZURE_DOC_INTELLIGENCE_ENDPOINT,
                api_key=config.AZURE_DOC_INTELLIGENCE_API_KEY,
                classifier_id=config.AZURE_DOC_INTELLIGENCE_CLASSIFIER_ID,
                api_version=config.AZURE_DOC_INTELLIGENCE_API_VERSION,
                timeout_sec=config.AZURE_DOC_INTELLIGENCE_TIMEOUT_SEC,
                poll_interval_sec=config.AZURE_DOC_INTELLIGENCE_POLL_INTERVAL_SEC,
            )
        LOGGER.warning("Azure classifier configured but missing endpoint/key/classifier id; using heuristic fallback")
    return HeuristicDocumentClassifier()


def classify_document(
    *,
    file_path: Path | None,
    filename: str,
    auto_tags: Iterable[str],
    text_samples: Iterable[str],
) -> ClassificationResult:
    classifier = build_document_classifier()
    return classifier.classify(
        file_path=file_path,
        filename=filename,
        auto_tags=auto_tags,
        text_samples=text_samples,
    )
