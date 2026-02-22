from .. import config

SYSTEM_PROMPT = (
    "You are a strict document QA assistant. You must answer the question using ONLY the provided context.\n"
    "Under NO circumstances should you use your external knowledge or hallucinate details.\n"
    "If the exact answer is not found in the context, explicitly state: 'I cannot find the answer in the provided documents.'\n"
    "Always cite your sources using the [source:...] tags provided in the text.\n"
    "Formatting rule: write in short paragraphs with a blank line between paragraphs. Do not use markdown bullets unless asked."
)

COMPARE_SYSTEM_PROMPT = (
    "You are a strict document comparison assistant. Your job is to compare evidence across multiple documents and time.\n"
    "You must answer using ONLY the provided context briefs and evidence.\n"
    "Under NO circumstances should you use external knowledge or make assumptions not explicitly written in the provided text.\n"
    "If the provided evidence is insufficient to make a comparison, explicitly state what is missing.\n"
    "Always cite your claims with the [source:...] tags.\n"
    "Formatting rules:\n"
    "- Put the direct answer first.\n"
    "- Explain changes over time in short paragraphs.\n"
    "- Use plain text section headers and paragraph spacing; avoid long numbered lists."
)


def build_prompt(question: str, context_blocks: list[str]) -> str:
    context = "\n".join(context_blocks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def build_compare_prompt(
    question: str,
    context_blocks: list[str],
    document_briefs: list[str],
    include_document_summaries: bool = True,
) -> str:
    context = "\n".join(context_blocks)
    briefs = "\n\n".join(document_briefs)
    summaries_instruction = (
        "3) Document summaries\n"
        "Place this section last. Use one short paragraph per document.\n\n"
        "Do not repeat the same document summary multiple times; merge duplicates or near-identical versions into one summary.\n\n"
        f"Document briefs:\n{briefs}\n\n"
    )
    if not include_document_summaries:
        summaries_instruction = (
            "3) Document summaries\n"
            "Do NOT include a document summaries section in your final answer.\n\n"
        )
    return (
        f"{COMPARE_SYSTEM_PROMPT}\n\n"
        "Required output order:\n"
        "1) Answer\n"
        "Write 1-2 short paragraphs that directly answer the question.\n\n"
        "2) Key changes over time\n"
        "Write short paragraphs describing notable additions/removals, role progression, and timeline shifts.\n\n"
        f"{summaries_instruction}"
        f"Evidence context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
