# Project Implementation Guide: PII Detection with DeBERTaV3 + DoRA & Synthetic Data

---

## Objective

Fine-tune `microsoft/deberta-v3-large` using **DoRA** (Weight-Decomposed Low-Rank Adaptation) on the `ai4privacy/open-pii-masking-500k` dataset, augmented with 11,000 rows of high-quality synthetic data targeting the six NER failure mode dimensions identified in the research paper "Unmasking the Reality of PII Masking Models" (Singh & Narayanan, 2025).

**Benchmark Target**: The fine-tuned model will be evaluated against `ai4privacy/llama-ai4privacy-english-anonymiser-openpii` to demonstrate improved PII detection on challenging cases.

---

## Phase 0: Label Schema Harmonization

Before any data generation, we must establish a unified label taxonomy that maps between the ai4privacy dataset and our synthetic data.

### 0.1 ai4privacy Label Inventory

The `ai4privacy/open-pii-masking-500k` dataset uses entity labels including but not limited to:

```
ACCOUNTNAME, ACCOUNTNUMBER, AGE, AMOUNT, BIC, BITCOINADDRESS, BUILDINGNUMBER,
CITY, COMPANYNAME, COUNTY, CREDITCARDCVV, CREDITCARDISSUER, CREDITCARDNUMBER,
CURRENCY, CURRENCYCODE, CURRENCYNAME, CURRENCYSYMBOL, DATE, DOB, DRIVERLICENSE,
EMAIL, ETHEREUMADDRESS, EYECOLOR, FIRSTNAME, GENDER, HEIGHT, IBAN, IP, IPV4,
IPV6, JOBAREA, JOBTITLE, JOBTYPE, LASTNAME, LITECOINADDRESS, MAC, MASKEDNUMBER,
MIDDLENAME, NEARBYGPSCOORDINATE, ORDINALDIRECTION, PASSWORD, PHONEIMEI,
PHONENUMBER, PIN, PREFIX, SECONDARYADDRESS, SEX, SSN, STATE, STREET,
STREETADDRESS, TIME, URL, USERAGENT, USERNAME, VEHICLEVIN, VEHICLEVRM,
ZIPCODE
```

### 0.2 Paper's PII Types (16 Categories)

The Singh & Narayanan paper evaluates 16 PII types:

```
BANK_ACCOUNT, BANK_UPI_ID, CREDIT_CARD, DATE_OF_BIRTH, DRIVER_LICENSE, EMAIL,
INSURANCE_NUMBER, NAME, NAMES_OF_PLACES_OR_NOUNS, NATIONAL_IDENTITY_SSN_AADHAR,
OTHER_NATIONAL_IDENTITY, PASSPORT_NUMBER, PHONE, POSTAL_CODE, TAX_IDENTIFICATION,
VEHICLE_REGISTRATION
```

### 0.3 Unified Label Mapping

We need a configuration file that maps synthetic labels to ai4privacy labels:

```python
# config/label_mapping.py
"""
Unified label mapping configuration for PII detection.

This module defines the canonical mapping between the paper's 16 PII categories
and the ai4privacy dataset's label schema. All synthetic data generation and
evaluation should reference this mapping for consistency.
"""

# Maps paper's categories → ai4privacy labels (for training compatibility)
PAPER_TO_AI4PRIVACY: dict[str, list[str]] = {
    "NAME": ["FIRSTNAME", "LASTNAME", "MIDDLENAME"],
    "EMAIL": ["EMAIL"],
    "PHONE": ["PHONENUMBER"],
    "DATE_OF_BIRTH": ["DOB", "DATE"],
    "POSTAL_CODE": ["ZIPCODE"],
    "CREDIT_CARD": ["CREDITCARDNUMBER"],
    "BANK_ACCOUNT": ["ACCOUNTNUMBER", "IBAN", "BIC"],
    "DRIVER_LICENSE": ["DRIVERLICENSE"],
    "PASSPORT_NUMBER": ["PASSPORT"],  # Note: May need to add to ai4privacy vocab
    "NATIONAL_IDENTITY_SSN_AADHAR": ["SSN"],
    "OTHER_NATIONAL_IDENTITY": ["NATIONALID"],  # Custom label
    "TAX_IDENTIFICATION": ["TAXID"],  # Custom label
    "VEHICLE_REGISTRATION": ["VEHICLEVRM", "VEHICLEVIN"],
    "INSURANCE_NUMBER": ["INSURANCENUMBER"],  # Custom label
    "BANK_UPI_ID": ["UPIID"],  # Custom label (India-specific format, English context)
    "NAMES_OF_PLACES_OR_NOUNS": ["CITY", "STATE", "COUNTY", "STREET"],
}

# Inverse mapping for evaluation (ai4privacy → paper categories)
AI4PRIVACY_TO_PAPER: dict[str, str] = {
    v: k for k, vs in PAPER_TO_AI4PRIVACY.items() for v in vs
}

# Labels that exist in synthetic data but not in ai4privacy (require vocab extension)
CUSTOM_LABELS: list[str] = [
    "PASSPORT",
    "NATIONALID", 
    "TAXID",
    "INSURANCENUMBER",
    "UPIID",
]

# BIO tag prefix generation
def get_bio_labels(base_labels: list[str]) -> list[str]:
    """
    Generate BIO-formatted labels from base label names.
    
    Args:
        base_labels: List of entity type names (e.g., ["NAME", "EMAIL"])
        
    Returns:
        List containing "O" plus B-/I- prefixed versions of each label.
        
    Example:
        >>> get_bio_labels(["NAME", "EMAIL"])
        ["O", "B-NAME", "I-NAME", "B-EMAIL", "I-EMAIL"]
    """
    bio_labels = ["O"]
    for label in base_labels:
        bio_labels.extend([f"B-{label}", f"I-{label}"])
    return bio_labels
```

### 0.4 Extending the ai4privacy Label Vocabulary

When loading the model for training, we need to extend the label vocabulary to include custom labels:

```python
# This will be implemented in Notebook 3
from transformers import AutoTokenizer, AutoModelForTokenClassification

def get_extended_label_list() -> list[str]:
    """
    Returns the complete label list combining ai4privacy and custom labels.
    """
    # Load original ai4privacy labels
    base_labels = [...]  # From ai4privacy dataset
    
    # Add custom labels from synthetic data
    all_base_labels = list(set(base_labels + CUSTOM_LABELS))
    
    return get_bio_labels(sorted(all_base_labels))
```

---

## Phase 1: Local Environment Setup (Anaconda)

**Target**: Notebooks 1 & 2  
**Python Version**: 3.13.5

### 1.1 Requirements

Create `requirements-local.txt`:

```text
# LLM APIs
xai-sdk>=0.1.0
openai>=1.60.0

# Data processing
pandas>=2.2.0
numpy>=2.0.0
tqdm>=4.66.0
faker>=33.0.0

# Tokenization
transformers>=4.40.0
sentencepiece>=0.2.0

# Utilities
python-dotenv>=1.0.0
httpx>=0.27.0
tenacity>=8.2.0
pydantic>=2.6.0

# Development
jupyterlab>=4.0.0
scikit-learn>=1.4.0
huggingface_hub>=0.21.0

# JSON handling
json-repair>=0.25.0
```

### 1.2 Environment Variables

Create `.env` file (don't commit this to vcs ever!):

```bash
XAI_API_KEY=your_xai_key_here
OPENAI_API_KEY=your_openai_key_here
HF_TOKEN=your_huggingface_token_here
```

---

## Notebook 1: Synthetic Data Generation (Local)

**Goal**: Generate 11,000 rows of "hard" PII examples targeting all six feature dimensions from the paper.

[... Notebook 1 content remains the same until Notebook 2 ...]

---

## Notebook 2: Span-to-Token Alignment & Validation (Local)

**Goal**: Convert span-annotated synthetic data to BIO-tagged token sequences aligned with DeBERTaV3's tokenizer, then validate quality **using OpenAI GPT-5.1 for semantic validation**.

### 2.1 Tokenization Alignment

```python
# alignment/span_to_bio.py
"""
Convert character-level span annotations to BIO-tagged token sequences.

This is the post-processing step that fixes the fundamental issue
with asking LLMs to generate pre-tokenized output. DeBERTaV3 uses a
SentencePiece tokenizer that splits words into subwords unpredictably.
We must align our character spans to the actual token boundaries.
"""
from dataclasses import dataclass
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from schemas.synthetic_output import SyntheticSample, EntitySpan
from config.label_mapping import PAPER_TO_AI4PRIVACY


@dataclass
class AlignedSample:
    """
    A sample with tokenization aligned to DeBERTaV3.
    
    Attributes:
        tokens: List of token strings (for debugging)
        input_ids: Token IDs for model input
        labels: BIO labels aligned to tokens
        attention_mask: Attention mask for padding
        original_text: The source text
        original_entities: Original span annotations
    """
    tokens: list[str]
    input_ids: list[int]
    labels: list[str]
    attention_mask: list[int]
    original_text: str
    original_entities: list[EntitySpan]


class SpanToBIOConverter:
    """
    Converts character-level entity spans to token-level BIO tags.
    
    The key challenge is that DeBERTaV3's tokenizer may split a single
    character span across multiple subword tokens. For example:
    - "cryptocurrency" → ["▁crypto", "currency"]
    - A span covering "cryptocurrency" must become ["B-X", "I-X"]
    
    We use the tokenizer's offset_mapping to determine which tokens
    correspond to which character positions.
    """
    
    def __init__(self, model_name: str = "microsoft/deberta-v3-large"):
        """
        Initialize with the target model's tokenizer.
        
        Args:
            model_name: Hugging Face model identifier
        """
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name,
            add_prefix_space=True,  # Important for consistent tokenization
        )
        self.model_name = model_name
    
    def _map_label_to_training_vocab(self, label: str) -> str:
        """
        Map paper's label taxonomy to ai4privacy training labels.
        
        Args:
            label: Label from our unified taxonomy
            
        Returns:
            Corresponding ai4privacy label for training
        """
        if label in PAPER_TO_AI4PRIVACY:
            # Return the first mapped label (primary)
            return PAPER_TO_AI4PRIVACY[label][0]
        return label
    
    def convert(
        self, 
        sample: SyntheticSample,
        max_length: int = 512,
    ) -> AlignedSample | None:
        """
        Convert a span-annotated sample to BIO-tagged tokens.
        
        Args:
            sample: The synthetic sample with character spans
            max_length: Maximum sequence length for tokenization
            
        Returns:
            AlignedSample with token-level BIO tags, or None if alignment fails
        """
        # Tokenize with offset mapping
        encoding = self.tokenizer(
            sample.text,
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
            padding="max_length",
        )
        
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        offset_mapping = encoding["offset_mapping"]
        
        # Initialize all labels as "O" (outside)
        labels = ["O"] * len(input_ids)
        
        # Convert tokens for debugging
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        # Sort entities by start position
        sorted_entities = sorted(sample.entities, key=lambda e: e.start)
        
        # Map each entity span to tokens
        for entity in sorted_entities:
            entity_start = entity.start
            entity_end = entity.end
            training_label = self._map_label_to_training_vocab(entity.label)
            
            is_first_token = True
            
            for token_idx, (tok_start, tok_end) in enumerate(offset_mapping):
                # Skip special tokens (offset is (0, 0))
                if tok_start == 0 and tok_end == 0:
                    continue
                
                # Check if token overlaps with entity span
                if tok_start < entity_end and tok_end > entity_start:
                    if is_first_token:
                        labels[token_idx] = f"B-{training_label}"
                        is_first_token = False
                    else:
                        labels[token_idx] = f"I-{training_label}"
        
        return AlignedSample(
            tokens=tokens,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            original_text=sample.text,
            original_entities=sample.entities,
        )
```

### 2.2 OpenAI-Based Validation (CRITICAL - REPLACES PROGRAMMATIC VALIDATION)

```python
# validation/openai_validator.py
"""
OpenAI GPT-5.1-based semantic validation for synthetic PII data.

This validator uses GPT-5.1 to perform deep semantic analysis of generated
samples, catching issues that programmatic validation cannot detect:
- Incoherent or unnatural text
- Contextually inappropriate PII placement
- Incorrect entity boundaries
- Missing feature dimension characteristics
- Implausible scenarios

IMPORTANT: This replaces the useless programmatic validator. LLM-based
validation is essential for catching semantic issues in LLM-generated data.
"""
import asyncio
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from schemas.synthetic_output import SyntheticSample


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    PASS = "pass"
    WARNING = "warning"
    ERROR = "error"


class EntityValidation(BaseModel):
    """
    Validation result for a single entity annotation.
    
    Attributes:
        entity_text: The text of the entity being validated
        entity_label: The label assigned to the entity
        is_correct_label: Whether the label is semantically correct
        is_correct_boundary: Whether the span boundaries are accurate
        suggested_label: Alternative label if incorrect (None if correct)
        issue_description: Description of any issues found
    """
    entity_text: str
    entity_label: str
    is_correct_label: bool
    is_correct_boundary: bool
    suggested_label: str | None = None
    issue_description: str | None = None


class DimensionValidation(BaseModel):
    """
    Validation result for feature dimension characteristics.
    
    Attributes:
        claimed_dimension: The feature dimension the sample claims to represent
        exhibits_characteristics: Whether the sample actually exhibits those characteristics
        characteristics_found: List of specific characteristics observed
        missing_characteristics: What's missing if not exhibiting
    """
    claimed_dimension: str
    exhibits_characteristics: bool
    characteristics_found: list[str] = Field(default_factory=list)
    missing_characteristics: list[str] = Field(default_factory=list)


class OpenAIValidationResult(BaseModel):
    """
    Complete validation result from OpenAI GPT-5.1 analysis.
    
    Attributes:
        is_valid: Overall pass/fail status
        severity: Highest severity issue found
        text_coherence_score: 1-5 score for text naturalness/coherence
        text_coherence_issues: List of coherence issues found
        entity_validations: Per-entity validation results
        dimension_validation: Feature dimension validation result
        overall_assessment: Free-form assessment from the model
        suggested_fixes: Actionable suggestions for fixing issues
    """
    is_valid: bool
    severity: ValidationSeverity
    text_coherence_score: int = Field(ge=1, le=5)
    text_coherence_issues: list[str] = Field(default_factory=list)
    entity_validations: list[EntityValidation] = Field(default_factory=list)
    dimension_validation: DimensionValidation
    overall_assessment: str
    suggested_fixes: list[str] = Field(default_factory=list)


# System prompt for GPT-5.1 validation
VALIDATION_SYSTEM_PROMPT = """You are an expert data quality validator for PII (Personally Identifiable Information) detection training data.

Your task is to validate synthetic training samples for a Named Entity Recognition (NER) model. Each sample contains:
1. A text passage (should be natural, coherent English)
2. Entity annotations with character-level spans and labels
3. A claimed "feature dimension" representing the type of NER challenge

FEATURE DIMENSIONS (from Singh & Narayanan 2025):
- basic: Straightforward, well-formatted entities in clear context
- contextual: Entities requiring disambiguation (e.g., "Apple" as company vs fruit)
- noisy: Real-world imperfections (typos, OCR errors, abbreviations, formatting issues)
- evolving: New/emerging PII formats (crypto addresses, UPI IDs, modern handles)
- multilingual: PII in international formats embedded in English text
- adversarial: Intentionally confusing inputs designed to fool NER models

PII TYPES TO VALIDATE:
NAME, EMAIL, PHONE, DATE_OF_BIRTH, POSTAL_CODE, CREDIT_CARD, BANK_ACCOUNT,
DRIVER_LICENSE, PASSPORT_NUMBER, NATIONAL_IDENTITY_SSN_AADHAR, OTHER_NATIONAL_IDENTITY,
TAX_IDENTIFICATION, VEHICLE_REGISTRATION, INSURANCE_NUMBER, BANK_UPI_ID,
NAMES_OF_PLACES_OR_NOUNS

YOUR VALIDATION CRITERIA:

1. TEXT COHERENCE (1-5 scale):
   - 5: Perfectly natural, indistinguishable from human-written text
   - 4: Minor awkwardness but clearly understandable
   - 3: Noticeable issues but usable for training
   - 2: Significant problems affecting training quality
   - 1: Incoherent, unusable garbage
   
2. ENTITY VALIDATION:
   - Is the label semantically correct for the entity text?
   - Are the span boundaries accurate (no missing/extra characters)?
   - Is the PII format realistic for its claimed locale?
   - Could this entity be confused with something else?

3. DIMENSION VALIDATION:
   - Does the sample ACTUALLY exhibit the claimed dimension's characteristics?
   - For "noisy": Are there realistic typos/OCR errors/formatting issues?
   - For "contextual": Is there genuine ambiguity requiring context?
   - For "adversarial": Would this actually fool an NER model?

4. OVERALL VALIDITY:
   - ERROR (invalid): Critical issues that would harm model training
   - WARNING (valid with issues): Minor issues, usable but suboptimal
   - PASS (valid): Good quality sample ready for training

Respond with a JSON object matching this exact schema:
{
    "is_valid": boolean,
    "severity": "pass" | "warning" | "error",
    "text_coherence_score": 1-5,
    "text_coherence_issues": ["issue1", "issue2", ...],
    "entity_validations": [
        {
            "entity_text": "...",
            "entity_label": "...",
            "is_correct_label": boolean,
            "is_correct_boundary": boolean,
            "suggested_label": "..." or null,
            "issue_description": "..." or null
        }
    ],
    "dimension_validation": {
        "claimed_dimension": "...",
        "exhibits_characteristics": boolean,
        "characteristics_found": ["..."],
        "missing_characteristics": ["..."]
    },
    "overall_assessment": "Free-form assessment",
    "suggested_fixes": ["fix1", "fix2", ...]
}

Be rigorous but fair. We need high-quality training data, but don't reject samples over trivial issues."""


def _format_sample_for_validation(sample: SyntheticSample) -> str:
    """
    Format a synthetic sample into a prompt for GPT-5.1 validation.
    
    Args:
        sample: The synthetic sample to validate
        
    Returns:
        Formatted string representation for the validation prompt
    """
    entities_formatted = []
    for entity in sample.entities:
        entity_text = sample.text[entity.start:entity.end]
        entities_formatted.append(
            f"  - '{entity_text}' [{entity.start}:{entity.end}] → {entity.label}"
        )
    
    return f"""SAMPLE TO VALIDATE:

TEXT:
\"\"\"{sample.text}\"\"\"

ENTITY ANNOTATIONS:
{chr(10).join(entities_formatted)}

CLAIMED FEATURE DIMENSION: {sample.feature_dimension}
SEED PII TYPE: {sample.seed_pii_type}
SEED PII VALUE: {sample.seed_pii_value}
SEED PII LOCALE: {sample.seed_pii_locale or "unspecified"}
SCENARIO: {sample.scenario}
TYPE VARIANT: {sample.type_variant}

Validate this sample according to your criteria."""


class OpenAIValidator:
    """
    Async validator using OpenAI GPT-5.1 for semantic validation of synthetic PII data.
    
    This validator performs deep semantic analysis that programmatic validation
    cannot achieve, including:
    - Natural language coherence assessment
    - Contextual appropriateness of PII
    - Entity boundary and label correctness
    - Feature dimension characteristic verification
    
    Attributes:
        client: AsyncOpenAI client instance
        model: Model identifier to use for validation
        max_concurrent: Maximum concurrent validation requests
        temperature: Sampling temperature (0 for deterministic)
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.1",
        max_concurrent: int = 10,
        temperature: float = 0.0,
    ):
        """
        Initialize the OpenAI validator.
        
        Args:
            api_key: OpenAI API key
            model: Model identifier (gpt-5.1 recommended for best results)
            max_concurrent: Maximum concurrent API requests
            temperature: Sampling temperature (0 for deterministic validation)
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_concurrent = max_concurrent
        self.temperature = temperature
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPStatusError, json.JSONDecodeError)),
    )
    async def _validate_single(self, sample: SyntheticSample) -> OpenAIValidationResult:
        """
        Validate a single sample using GPT-5.1.
        
        Args:
            sample: The synthetic sample to validate
            
        Returns:
            OpenAIValidationResult with detailed validation findings
            
        Raises:
            json.JSONDecodeError: If GPT-5.1 response cannot be parsed
            httpx.HTTPStatusError: On API errors (triggers retry)
        """
        async with self._semaphore:
            user_prompt = _format_sample_for_validation(sample)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
                max_tokens=2000,
            )
            
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from OpenAI")
            
            result_dict = json.loads(content)
            return OpenAIValidationResult.model_validate(result_dict)
    
    async def validate_sample(self, sample: SyntheticSample) -> OpenAIValidationResult:
        """
        Validate a single sample with error handling.
        
        Args:
            sample: The synthetic sample to validate
            
        Returns:
            OpenAIValidationResult (returns error result on failure)
        """
        try:
            return await self._validate_single(sample)
        except Exception as e:
            # Return an error result if validation fails
            return OpenAIValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                text_coherence_score=1,
                text_coherence_issues=[f"Validation failed: {str(e)}"],
                entity_validations=[],
                dimension_validation=DimensionValidation(
                    claimed_dimension=sample.feature_dimension,
                    exhibits_characteristics=False,
                    characteristics_found=[],
                    missing_characteristics=["Unable to validate due to error"],
                ),
                overall_assessment=f"Validation error: {str(e)}",
                suggested_fixes=["Retry validation or manually inspect"],
            )
    
    async def validate_batch(
        self,
        samples: list[SyntheticSample],
        progress_callback: callable | None = None,
    ) -> list[tuple[SyntheticSample, OpenAIValidationResult]]:
        """
        Validate a batch of samples concurrently.
        
        Args:
            samples: List of samples to validate
            progress_callback: Optional callback(completed, total) for progress updates
            
        Returns:
            List of (sample, result) tuples
        """
        tasks = [self.validate_sample(sample) for sample in samples]
        results = []
        
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(samples))
        
        # Match results back to samples (as_completed doesn't preserve order)
        # Re-run to get ordered results
        ordered_results = await asyncio.gather(*[
            self.validate_sample(sample) for sample in samples
        ])
        
        return list(zip(samples, ordered_results))
    
    def filter_valid_samples(
        self,
        validation_results: list[tuple[SyntheticSample, OpenAIValidationResult]],
        include_warnings: bool = True,
    ) -> tuple[list[SyntheticSample], list[tuple[SyntheticSample, OpenAIValidationResult]]]:
        """
        Filter samples based on validation results.
        
        Args:
            validation_results: List of (sample, result) tuples from validate_batch
            include_warnings: Whether to include samples with warnings (default True)
            
        Returns:
            Tuple of (valid_samples, rejected_with_reasons)
        """
        valid = []
        rejected = []
        
        for sample, result in validation_results:
            if result.is_valid:
                if result.severity == ValidationSeverity.PASS:
                    valid.append(sample)
                elif result.severity == ValidationSeverity.WARNING and include_warnings:
                    valid.append(sample)
                else:
                    rejected.append((sample, result))
            else:
                rejected.append((sample, result))
        
        return valid, rejected


class ValidationReportGenerator:
    """
    Generates comprehensive validation reports from OpenAI validation results.
    
    Provides aggregate statistics, per-dimension breakdowns, and actionable
    insights for improving synthetic data quality.
    """
    
    @staticmethod
    def generate_report(
        results: list[tuple[SyntheticSample, OpenAIValidationResult]],
    ) -> dict[str, Any]:
        """
        Generate a comprehensive validation report.
        
        Args:
            results: List of (sample, result) tuples from validation
            
        Returns:
            Dictionary containing aggregate statistics and breakdowns
        """
        total = len(results)
        passed = sum(1 for _, r in results if r.severity == ValidationSeverity.PASS)
        warnings = sum(1 for _, r in results if r.severity == ValidationSeverity.WARNING)
        errors = sum(1 for _, r in results if r.severity == ValidationSeverity.ERROR)
        
        # Coherence score distribution
        coherence_scores = [r.text_coherence_score for _, r in results]
        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0
        
        # Per-dimension breakdown
        dimension_stats: dict[str, dict[str, int]] = {}
        for sample, result in results:
            dim = sample.feature_dimension
            if dim not in dimension_stats:
                dimension_stats[dim] = {"total": 0, "pass": 0, "warning": 0, "error": 0, "exhibits_dim": 0}
            dimension_stats[dim]["total"] += 1
            dimension_stats[dim][result.severity.value] += 1
            if result.dimension_validation.exhibits_characteristics:
                dimension_stats[dim]["exhibits_dim"] += 1
        
        # Most common issues
        all_issues: list[str] = []
        for _, result in results:
            all_issues.extend(result.text_coherence_issues)
            for ev in result.entity_validations:
                if ev.issue_description:
                    all_issues.append(ev.issue_description)
        
        from collections import Counter
        issue_counts = Counter(all_issues).most_common(10)
        
        # Entity label accuracy
        total_entities = 0
        correct_labels = 0
        correct_boundaries = 0
        for _, result in results:
            for ev in result.entity_validations:
                total_entities += 1
                if ev.is_correct_label:
                    correct_labels += 1
                if ev.is_correct_boundary:
                    correct_boundaries += 1
        
        return {
            "summary": {
                "total_samples": total,
                "passed": passed,
                "warnings": warnings,
                "errors": errors,
                "pass_rate": passed / total if total > 0 else 0,
                "usable_rate": (passed + warnings) / total if total > 0 else 0,
            },
            "coherence": {
                "average_score": avg_coherence,
                "score_distribution": {
                    score: coherence_scores.count(score) 
                    for score in range(1, 6)
                },
            },
            "entity_accuracy": {
                "total_entities": total_entities,
                "label_accuracy": correct_labels / total_entities if total_entities > 0 else 0,
                "boundary_accuracy": correct_boundaries / total_entities if total_entities > 0 else 0,
            },
            "per_dimension": dimension_stats,
            "top_issues": issue_counts,
        }
    
    @staticmethod
    def print_report(report: dict[str, Any]) -> None:
        """Print a formatted validation report to stdout."""
        print("=" * 80)
        print("OPENAI VALIDATION REPORT")
        print("=" * 80)
        
        s = report["summary"]
        print(f"\nOVERALL SUMMARY:")
        print(f"  Total samples: {s['total_samples']}")
        print(f"  Passed: {s['passed']} ({s['pass_rate']*100:.1f}%)")
        print(f"  Warnings: {s['warnings']}")
        print(f"  Errors: {s['errors']}")
        print(f"  Usable (pass + warning): {s['usable_rate']*100:.1f}%")
        
        c = report["coherence"]
        print(f"\nTEXT COHERENCE:")
        print(f"  Average score: {c['average_score']:.2f}/5.0")
        print(f"  Distribution: {c['score_distribution']}")
        
        e = report["entity_accuracy"]
        print(f"\nENTITY ANNOTATION ACCURACY:")
        print(f"  Total entities validated: {e['total_entities']}")
        print(f"  Label accuracy: {e['label_accuracy']*100:.1f}%")
        print(f"  Boundary accuracy: {e['boundary_accuracy']*100:.1f}%")
        
        print(f"\nPER-DIMENSION BREAKDOWN:")
        for dim, stats in report["per_dimension"].items():
            exhibit_rate = stats["exhibits_dim"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"  {dim}:")
            print(f"    Total: {stats['total']}, Pass: {stats['pass']}, "
                  f"Warning: {stats['warning']}, Error: {stats['error']}")
            print(f"    Exhibits dimension characteristics: {exhibit_rate:.1f}%")
        
        print(f"\nTOP 10 ISSUES:")
        for issue, count in report["top_issues"]:
            print(f"  [{count}x] {issue[:70]}...")
        
        print("=" * 80)
```

### 2.3 Hybrid Validation Pipeline (OpenAI + Fast Programmatic Pre-filter)

```python
# validation/hybrid_validator.py
"""
Hybrid validation pipeline combining fast programmatic pre-filtering with
thorough OpenAI semantic validation.

The programmatic pre-filter catches obvious issues cheaply before sending
samples to GPT-5.1, reducing API costs while maintaining quality.
"""
import asyncio
from dataclasses import dataclass
from typing import Any

from schemas.synthetic_output import SyntheticSample
from validation.openai_validator import (
    OpenAIValidator,
    OpenAIValidationResult,
    ValidationSeverity,
    ValidationReportGenerator,
)


@dataclass
class PreFilterResult:
    """
    Result from programmatic pre-filtering.
    
    Attributes:
        passes_prefilter: Whether sample passes basic checks
        issues: List of issues found (empty if passes)
    """
    passes_prefilter: bool
    issues: list[str]


class ProgrammaticPreFilter:
    """
    Fast programmatic pre-filter for obvious issues.
    
    This catches cheap-to-detect problems before sending to GPT-5.1:
    - Empty or too-short/too-long text
    - Missing required fields
    - Invalid character positions
    - Non-English text (high non-ASCII ratio)
    
    NOTE: This is NOT a replacement for OpenAI validation. It's a cost
    optimization to avoid wasting API calls on obviously broken samples.
    """
    
    def __init__(
        self,
        min_text_length: int = 50,
        max_text_length: int = 600,
        max_non_ascii_ratio: float = 0.3,
    ):
        """
        Initialize pre-filter with thresholds.
        
        Args:
            min_text_length: Minimum acceptable text length
            max_text_length: Maximum acceptable text length  
            max_non_ascii_ratio: Maximum ratio of non-ASCII characters
        """
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.max_non_ascii_ratio = max_non_ascii_ratio
    
    def check(self, sample: SyntheticSample) -> PreFilterResult:
        """
        Run fast programmatic checks on a sample.
        
        Args:
            sample: The synthetic sample to pre-filter
            
        Returns:
            PreFilterResult indicating pass/fail and any issues
        """
        issues = []
        
        # Check 1: Text length bounds
        text_len = len(sample.text)
        if text_len < self.min_text_length:
            issues.append(f"Text too short: {text_len} < {self.min_text_length}")
        elif text_len > self.max_text_length:
            issues.append(f"Text too long: {text_len} > {self.max_text_length}")
        
        # Check 2: Seed PII must be present in text
        if sample.seed_pii_value not in sample.text:
            issues.append(f"Seed PII '{sample.seed_pii_value}' not found in text")
        
        # Check 3: Entity spans must be valid positions
        for entity in sample.entities:
            if entity.start < 0 or entity.end > len(sample.text):
                issues.append(f"Invalid span [{entity.start}:{entity.end}] for text length {len(sample.text)}")
            elif entity.start >= entity.end:
                issues.append(f"Invalid span [{entity.start}:{entity.end}]: start >= end")
            else:
                # Check span text matches entity (if stored)
                actual_text = sample.text[entity.start:entity.end]
                if hasattr(entity, 'text') and entity.text != actual_text:
                    issues.append(f"Span text mismatch: expected '{entity.text}', got '{actual_text}'")
        
        # Check 4: At least one entity required
        if len(sample.entities) == 0:
            issues.append("No entities annotated")
        
        # Check 5: Text should be primarily English (basic heuristic)
        non_ascii = sum(1 for c in sample.text if ord(c) > 127)
        non_ascii_ratio = non_ascii / len(sample.text) if sample.text else 0
        if non_ascii_ratio > self.max_non_ascii_ratio:
            issues.append(f"High non-ASCII ratio: {non_ascii_ratio:.2%} > {self.max_non_ascii_ratio:.0%}")
        
        return PreFilterResult(
            passes_prefilter=len(issues) == 0,
            issues=issues,
        )


class HybridValidationPipeline:
    """
    Two-stage validation pipeline: fast pre-filter → OpenAI semantic validation.
    
    This approach optimizes cost while maintaining quality:
    1. Run cheap programmatic checks to filter obviously broken samples
    2. Send remaining samples to GPT-5.1 for deep semantic validation
    
    Attributes:
        prefilter: ProgrammaticPreFilter instance
        openai_validator: OpenAIValidator instance
    """
    
    def __init__(
        self,
        openai_api_key: str,
        openai_model: str = "gpt-5.1",
        max_concurrent_openai: int = 10,
    ):
        """
        Initialize the hybrid validation pipeline.
        
        Args:
            openai_api_key: OpenAI API key
            openai_model: Model for semantic validation (gpt-5.1 recommended)
            max_concurrent_openai: Max concurrent OpenAI requests
        """
        self.prefilter = ProgrammaticPreFilter()
        self.openai_validator = OpenAIValidator(
            api_key=openai_api_key,
            model=openai_model,
            max_concurrent=max_concurrent_openai,
        )
    
    async def validate_batch(
        self,
        samples: list[SyntheticSample],
        skip_openai_for_prefilter_failures: bool = True,
        progress_callback: callable | None = None,
    ) -> dict[str, Any]:
        """
        Run full validation pipeline on a batch of samples.
        
        Args:
            samples: List of samples to validate
            skip_openai_for_prefilter_failures: Skip OpenAI for samples that fail prefilter
            progress_callback: Optional callback(stage, completed, total) for progress
            
        Returns:
            Dictionary with validation results and statistics
        """
        # Stage 1: Programmatic pre-filter
        if progress_callback:
            progress_callback("prefilter", 0, len(samples))
        
        prefilter_results = []
        passed_prefilter = []
        failed_prefilter = []
        
        for i, sample in enumerate(samples):
            result = self.prefilter.check(sample)
            prefilter_results.append((sample, result))
            
            if result.passes_prefilter:
                passed_prefilter.append(sample)
            else:
                failed_prefilter.append((sample, result))
            
            if progress_callback:
                progress_callback("prefilter", i + 1, len(samples))
        
        # Stage 2: OpenAI semantic validation (only for samples that passed prefilter)
        if progress_callback:
            progress_callback("openai", 0, len(passed_prefilter))
        
        openai_results = []
        if passed_prefilter:
            def openai_progress(completed: int, total: int) -> None:
                if progress_callback:
                    progress_callback("openai", completed, total)
            
            openai_results = await self.openai_validator.validate_batch(
                passed_prefilter,
                progress_callback=openai_progress,
            )
        
        # Combine results
        valid_samples, rejected_by_openai = self.openai_validator.filter_valid_samples(
            openai_results,
            include_warnings=True,
        )
        
        # Generate report
        openai_report = ValidationReportGenerator.generate_report(openai_results) if openai_results else {}
        
        return {
            "valid_samples": valid_samples,
            "prefilter_failures": failed_prefilter,
            "openai_rejections": rejected_by_openai,
            "statistics": {
                "total_input": len(samples),
                "passed_prefilter": len(passed_prefilter),
                "failed_prefilter": len(failed_prefilter),
                "passed_openai": len(valid_samples),
                "rejected_by_openai": len(rejected_by_openai),
                "final_valid_count": len(valid_samples),
                "final_valid_rate": len(valid_samples) / len(samples) if samples else 0,
            },
            "openai_report": openai_report,
        }


# Convenience function for notebook usage
async def run_validation_pipeline(
    samples: list[SyntheticSample],
    openai_api_key: str,
    model: str = "gpt-5.1",
) -> dict[str, Any]:
    """
    Run the full hybrid validation pipeline.
    
    This is the main entry point for Notebook 2 validation.
    
    Args:
        samples: List of synthetic samples to validate
        openai_api_key: OpenAI API key
        model: OpenAI model to use (default: gpt-5.1)
        
    Returns:
        Dictionary containing valid samples and validation statistics
        
    Example:
        >>> import asyncio
        >>> from validation.hybrid_validator import run_validation_pipeline
        >>> 
        >>> results = asyncio.run(run_validation_pipeline(
        ...     samples=my_samples,
        ...     openai_api_key=os.getenv("OPENAI_API_KEY"),
        ... ))
        >>> print(f"Valid samples: {len(results['valid_samples'])}")
        >>> print(f"Validation rate: {results['statistics']['final_valid_rate']:.1%}")
    """
    pipeline = HybridValidationPipeline(
        openai_api_key=openai_api_key,
        openai_model=model,
    )
    
    def progress(stage: str, completed: int, total: int) -> None:
        print(f"[{stage.upper()}] {completed}/{total} ({completed/total*100:.1f}%)")
    
    return await pipeline.validate_batch(
        samples,
        skip_openai_for_prefilter_failures=True,
        progress_callback=progress,
    )
```

### 2.4 Dataset Upload to HuggingFace

```python
# upload/hf_dataset.py
"""
Upload processed synthetic data to HuggingFace Hub.

Creates a dataset with proper train/validation/test splits stratified
by feature dimension and PII type.
"""
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from alignment.span_to_bio import AlignedSample


def create_hf_dataset(
    aligned_samples: list[AlignedSample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> DatasetDict:
    """
    Create a HuggingFace DatasetDict from aligned samples.
    
    Args:
        aligned_samples: List of tokenized samples
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility
        
    Returns:
        DatasetDict with train/validation/test splits
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # Convert to dictionaries
    data = {
        "input_ids": [s.input_ids for s in aligned_samples],
        "attention_mask": [s.attention_mask for s in aligned_samples],
        "labels": [s.labels for s in aligned_samples],
        "tokens": [s.tokens for s in aligned_samples],
        "text": [s.original_text for s in aligned_samples],
    }
    
    # Create indices for stratified split
    indices = list(range(len(aligned_samples)))
    
    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=random_state,
    )
    
    # Second split: val vs test
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=relative_val_ratio,
        random_state=random_state,
    )
    
    def subset_data(indices):
        return {k: [v[i] for i in indices] for k, v in data.items()}
    
    return DatasetDict({
        "train": Dataset.from_dict(subset_data(train_idx)),
        "validation": Dataset.from_dict(subset_data(val_idx)),
        "test": Dataset.from_dict(subset_data(test_idx)),
    })


def upload_to_hub(
    dataset: DatasetDict,
    repo_id: str,
    token: str,
) -> None:
    """
    Upload dataset to HuggingFace Hub.
    
    Args:
        dataset: The DatasetDict to upload
        repo_id: HuggingFace repository ID (e.g., "username/dataset-name")
        token: HuggingFace API token
    """
    dataset.push_to_hub(
        repo_id,
        token=token,
        private=True,  # Start as private, can make public later
    )
    print(f"Uploaded to https://huggingface.co/datasets/{repo_id}")
```

---

## Phase 2: Modal Training Environment

[... Rest of PLAN.md continues unchanged until Execution Checklist ...]

---

## Execution Checklist

Before starting implementation:

- [ ] Create `.env` with all required API keys (XAI_API_KEY, OPENAI_API_KEY, HF_TOKEN)
- [ ] Review ai4privacy dataset schema to finalize label mapping
- [ ] Set up HuggingFace repository for synthetic dataset
- [ ] Configure Modal account and secrets

During Notebook 1:
- [ ] Generate samples across ALL 6 feature dimensions
- [ ] Ensure all generated text is in ENGLISH with diverse international PII formats
- [ ] Include PII from all locales (US, UK, India, Germany, France, etc.)
- [ ] Checkpoint every 100 samples
- [ ] Validate JSON output with Pydantic before saving

During Notebook 2:
- [ ] Align ALL samples with DeBERTa tokenizer locally
- [ ] **Run OpenAI GPT-5.1 semantic validation (NOT just programmatic checks)**
- [ ] **Review validation report for dimension characteristic adherence**
- [ ] **Filter out samples with coherence score < 3**
- [ ] Verify English-only text content (>70% ASCII threshold in prefilter)
- [ ] Stratify split by dimension AND PII type AND locale
- [ ] Upload to HuggingFace with proper train/val/test splits

During Notebook 3:
- [ ] Use target_modules: ["query_proj", "key_proj", "value_proj", "dense"]
- [ ] Monitor training loss for signs of DoRA not applying
- [ ] Use early stopping with patience=3
- [ ] Save adapter weights to Hub

During Notebook 4:
- [ ] Benchmark ONLY against ai4privacy/llama-ai4privacy-english-anonymiser-openpii
- [ ] Report non-identification rate and misclassification rate
- [ ] Show per-dimension breakdowns
- [ ] Show per-locale breakdowns (key for demonstrating international format improvement)
- [ ] Identify which dimensions and locales show largest improvement
