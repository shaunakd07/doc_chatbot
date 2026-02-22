from .. import config

SYSTEM_PROMPT = (
    "You are a strict document QA assistant. You must answer the question using ONLY the provided context.\n"
    "Under NO circumstances should you use your external knowledge or hallucinate details.\n"
    "If the exact answer is not found in the context, explicitly state: 'I cannot find the answer in the provided documents.'\n"
    "Always cite your sources using the [source:...] tags provided in the text."
)

COMPARE_SYSTEM_PROMPT = (
    "You are a strict document comparison assistant. Your job is to compare evidence across multiple documents and time.\n"
    "You must answer using ONLY the provided context briefs and evidence.\n"
    "Under NO circumstances should you use external knowledge or make assumptions not explicitly written in the provided text.\n"
    "If the provided evidence is insufficient to make a comparison, explicitly state what is missing.\n"
    "Always cite your claims with the [source:...] tags."
)


def build_prompt(question: str, context_blocks: list[str]) -> str:
    context = "\n".join(context_blocks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def build_compare_prompt(question: str, context_blocks: list[str], document_briefs: list[str]) -> str:
    context = "\n".join(context_blocks)
    briefs = "\n\n".join(document_briefs)
    return (
        f"{COMPARE_SYSTEM_PROMPT}\n\n"
        "Task:\n"
        "- Summarize each document briefly.\n"
        "- Identify differences over time and notable additions/removals.\n"
        "- End with a concise 'Bottom line' section.\n\n"
        f"Document briefs:\n{briefs}\n\n"
        f"Evidence context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
