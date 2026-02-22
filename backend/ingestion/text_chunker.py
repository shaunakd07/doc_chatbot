from typing import Iterable, List


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + max_chars)
        if end < len(cleaned):
            space_idx = cleaned.rfind(" ", start, end)
            if space_idx > start + int(max_chars * 0.6):
                end = space_idx
        chunk = cleaned[start:end]
        chunks.append(chunk)
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
        if start > 0 and cleaned[start] != " ":
            next_space = cleaned.find(" ", start)
            if next_space != -1:
                start = next_space + 1
        while start < len(cleaned) and cleaned[start] == " ":
            start += 1
    return chunks


def chunk_many(texts: Iterable[str], max_chars: int = 1000, overlap: int = 200) -> List[str]:
    out: List[str] = []
    for text in texts:
        out.extend(chunk_text(text, max_chars=max_chars, overlap=overlap))
    return out
