# Document-Type Classification Research (2026-03-03)

> Status note: this file is a research snapshot, not a statement of the current default runtime configuration.
>
> Current implementation status in the repo:
> - default classifier provider: `heuristic`
> - optional implemented providers: `semantic_openai`, `azure_document_intelligence`
> - not implemented in this repo: Google Cloud Document AI, AWS Textract + Comprehend orchestration, self-hosted LayoutLM serving/training

## Goal

Rank practical options for classifying enterprise documents and supporting out-of-place detection in labeled folders with high precision.

## Current repo mapping

What the repo actually does today:

- baseline ingestion writes `doc_type`, `doc_type_confidence`, and `doc_type_scores`
- the backend can backfill or refine missing predictions through the provider abstraction in `backend/services/document_classifier.py`
- folder-level review logic uses those predictions in `backend/services/out_of_place_detection.py`
- `.env.example` currently sets `DOC_TYPE_CLASSIFIER_PROVIDER=heuristic`

So this research remains relevant for future provider decisions, but the production-default path in the current repo is still heuristic classification.

## Ranked Options

### 1. Azure AI Document Intelligence

Why it ranked first:

- strong fit for enterprise governance and confidence-scored outputs
- direct Python and REST integration into the current backend shape
- good match for advisory review workflows

Trade-offs:

- requires labeled training data
- introduces managed-service cost and operational setup

Current repo status:

- provider is implemented as `azure_document_intelligence`
- it is optional, not default
- it requires the `AZURE_DOC_INTELLIGENCE_*` settings to be present

Primary sources:

- Custom classification overview: https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/train/custom-classifier?view=doc-intel-4.0.0
- Data/privacy posture: https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/faq?view=doc-intel-4.0.0
- Service limits/performance constraints: https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/service-limits?view=doc-intel-4.0.0
- Pricing model: https://azure.microsoft.com/en-us/pricing/details/ai-document-intelligence/

### 2. Google Cloud Document AI

Why it ranked highly:

- mature managed document stack
- good fit for GCP-native workflows

Trade-offs:

- not integrated into the current backend
- would require new provider implementation and operational setup

Current repo status:

- research only
- not implemented

Primary sources:

- Custom classifier setup: https://cloud.google.com/document-ai/docs/custom-classifier
- Pricing: https://cloud.google.com/document-ai/pricing
- SLA: https://cloud.google.com/document-ai/sla
- Security and data handling overview: https://cloud.google.com/security/privacy

### 3. AWS Textract + Amazon Comprehend Custom Classification

Why it ranked third:

- flexible AWS-native path
- good fit if the rest of the stack is already AWS-centric

Trade-offs:

- more assembly work than a single managed classifier
- multi-service orchestration complexity

Current repo status:

- research only
- not implemented

Primary sources:

- Comprehend custom classification: https://docs.aws.amazon.com/comprehend/latest/dg/how-document-classification.html
- Textract APIs and capabilities: https://docs.aws.amazon.com/textract/latest/dg/API_Operations_Amazon_Textract_Service.html
- AWS compliance program reference: https://aws.amazon.com/compliance/programs/
- Textract data protection/compliance references: https://docs.aws.amazon.com/textract/latest/dg/data-protection.html

### 4. Self-hosted layout-aware models

Why it remained interesting:

- strongest control over data boundary and serving behavior
- avoids managed-service dependency

Trade-offs:

- highest MLOps burden
- training, drift, serving, and governance are all on the team

Current repo status:

- not implemented
- would require a substantially different operational model than the current OpenAI-plus-services architecture

Primary sources:

- LayoutLMv3 paper: https://arxiv.org/abs/2204.08387
- Hugging Face document classification task support: https://huggingface.co/docs/transformers/tasks/sequence_classification

## Recommendation at the time

The original recommendation was:

- primary enterprise option: Azure Document Intelligence custom classifier
- resilience fallback: local heuristic classifier

## Current recommendation for this repo

For the codebase as it exists today:

1. keep `heuristic` as the safest default for development and low-friction environments
2. use `semantic_openai` when you want better label flexibility without taking on Azure model lifecycle work
3. use `azure_document_intelligence` only for tenants or deployments that specifically need managed document-classification controls and are prepared to configure and operate that path

That recommendation matches the current provider implementation and `.env.example` defaults more closely than the original memo did.
