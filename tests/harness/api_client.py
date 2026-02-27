from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


class ApiClient:
    def __init__(self, base_url: str, timeout_sec: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_sec),
            headers={"Accept": "application/json"},
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "ApiClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _parse_json(self, response: httpx.Response) -> Dict[str, Any]:
        try:
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"Non-JSON response ({response.status_code}) from {response.request.url}: "
                f"{response.text[:400]}"
            ) from exc
        if isinstance(payload, dict):
            return payload
        return {"raw": payload}

    def _request(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        response = self.client.request(method=method, url=path, **kwargs)
        payload = self._parse_json(response)
        if response.status_code >= 400:
            message = payload.get("error") if isinstance(payload, dict) else None
            raise RuntimeError(
                f"API {method} {path} failed ({response.status_code}): "
                f"{message or json.dumps(payload)[:400]}"
            )
        return payload

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/api/health")

    def list_documents(self) -> List[Dict[str, Any]]:
        payload = self._request("GET", "/api/documents")
        records = payload.get("documents")
        if not isinstance(records, list):
            return []
        out: List[Dict[str, Any]] = []
        for row in records:
            if isinstance(row, dict):
                out.append(row)
        return out

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/api/documents/{doc_id}")

    def get_document_graphs(self, doc_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/api/documents/{doc_id}/diagram-graphs")

    def upload_document(self, file_path: Path) -> Dict[str, Any]:
        with file_path.open("rb") as handle:
            files = {"file": (file_path.name, handle, "application/octet-stream")}
            return self._request("POST", "/api/documents", files=files)

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/api/documents/{doc_id}")

    def delete_all_documents(self) -> Dict[str, Any]:
        return self._request("DELETE", "/api/documents")

    def chat(
        self,
        message: str,
        *,
        doc_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        include_document_summaries: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "message": str(message),
            "doc_ids": doc_ids,
            "top_k": top_k,
            "include_document_summaries": bool(include_document_summaries),
        }
        return self._request("POST", "/api/chat", json=payload)

