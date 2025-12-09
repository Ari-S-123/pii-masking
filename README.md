# Improving PII Detection with DeBERTaV3 + DoRA & Synthetic Data

Fine-tuning DeBERTaV3-large using Weight-Decomposed Low-Rank Adaptation (DoRA) on synthetic data targeting NER failure modes for robust Personally Identifiable Information detection.

**Authors:** Aritra Saharay & Jonathan Tarun Rajasekaran  
**Institution:** Northeastern University  
**Course:** CS 6140 Machine Learning — Final Project  
**Date:** December 2025

---

## Abstract

Personally Identifiable Information (PII) detection is critical for data privacy compliance, yet existing open-source models exhibit significant performance gaps on real-world data. Singh & Narayanan (2025) demonstrated that popular models like Piiranha and Starpii fail to detect PII in 28% of cases and misclassify entities 67% of the time. We address these gaps by fine-tuning DeBERTaV3-large using Weight-Decomposed Low-Rank Adaptation (DoRA) on a combined dataset of 156K examples, including 6,002 synthetically generated samples specifically targeting six NER failure mode dimensions. Our model achieves an F1 score of **0.908**, outperforming the best existing open-source model (Piiranha, F1=0.244) by **272%** while supporting 505 entity labels across 10 regional PII formats.

---

## Motivation

Privacy masking is a fundamental requirement for organizations handling sensitive data under regulations such as GDPR and CCPA. Named Entity Recognition (NER) approaches are widely used for PII detection, but they face significant challenges including content ambiguity, regional format variations, and adversarial inputs.

Singh & Narayanan (2025) evaluated three prominent PII masking models across 17K test samples and found:

- **28%** complete non-identification of any PII
- **67%** misclassification of detected entities
- Particularly poor performance on non-US regional formats

These failures stem from sparse training data, lack of regional diversity, and insufficient coverage of emerging PII types like cryptocurrency wallets and UPI identifiers.

---

## Approach

Our objective was to build a more robust PII detection model by:

1. Combining the ai4privacy English dataset with synthetically generated challenging examples
2. Using DoRA fine-tuning for parameter-efficient adaptation
3. Specifically targeting the six NER failure dimensions identified in prior work

---

## Dataset

Our training dataset combines two sources into **156,695 total examples**:

| Data Source | Train | Test | Description |
|-------------|-------|------|-------------|
| ai4privacy (English subset) | 120,533 | 30,160 | Existing open-source PII dataset |
| Synthetic (Grok 4.1 + GPT-5.1) | 4,801 | 1,201 | Generated targeting NER failure modes |
| **Total** | **125,334** | **31,361** | — |

### Synthetic Data Generation

We generated synthetic samples using Grok (xAI) with GPT-5.1 validation, targeting six NER failure dimensions from Singh & Narayanan's framework:

| Dimension | Count (After Validation) | Description |
|-----------|--------------------------|-------------|
| Basic | 989 | Standard, well-formatted entities |
| Contextual | 801 | Ambiguous entities requiring context |
| Noisy | 876 | Typos, informal text, OCR-like errors |
| Evolving | 754 | Modern PII: crypto wallets, UPI, TOTP |
| Multilingual | 917 | International formats in English text |
| Adversarial | 464 | Intentionally deceptive patterns |

**Generation Pipeline:** Of 10,944 generation attempts, Grok successfully produced 10,836 samples (99.0%). After GPT-5.1 validation filtering for coherence, boundary accuracy, and label correctness, 6,002 samples (55.4%) were retained. Entity boundary accuracy was 96.6%, while label accuracy was 49.3% — the latter primarily due to schema mismatches (e.g., "PERSON" vs "NAME") resolved through post-processing.

**Regional Coverage:** Samples cover 10 locales (India, US, UK, Canada, Australia, Germany, France, Spain, Italy, Netherlands) with 16 PII types from Singh & Narayanan's taxonomy, plus emerging types like cryptocurrency addresses and UPI IDs.

---

## Methodology

### Model Architecture

We fine-tune `microsoft/deberta-v3-large` (304M parameters) for token classification using Weight-Decomposed Low-Rank Adaptation (DoRA). DoRA decomposes pretrained weights into magnitude and direction components, applying low-rank updates only to the directional component. This approach has shown superior performance over standard LoRA for complex NLP tasks while maintaining parameter efficiency.

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Base Model | `microsoft/deberta-v3-large` |
| Learning Rate | 2e-5 |
| Epochs | 5 (with early stopping) |
| Batch Size | 32 |
| DoRA Rank (r) | 16 |
| DoRA Alpha | 32 |
| DoRA Dropout | 0.05 |
| Target Modules | query_proj, key_proj, value_proj, dense |
| Trainable Parameters | 7.82M (2.6% of total) |
| Hardware | NVIDIA H100 (Modal) |

**Preventing Overfitting:** We employed early stopping with patience=3 based on validation F1 score, DoRA regularization (dropout=0.05), and stratified train/test splits. Training completed at 19,000 steps with final training loss of 0.271.

---

## Results

### Model Comparison

We evaluated our model against four existing PII detection models on the held-out test set (31,361 samples):

| Rank | Model | Precision | Recall | F1-Score | Label Count |
|------|-------|-----------|--------|----------|-------------|
| 1 | **DeBERTa-v3-large + DoRA (Ours)** | **0.898** | **0.918** | **0.908** | **505** |
| 2 | mDeBERTa-v3-base (ai4privacy) | 0.370 | 0.369 | 0.370 | 111 |
| 3 | DistilBERT (ai4privacy) | 0.342 | 0.368 | 0.355 | 116 |
| 4 | Piiranha v1 | 0.600 | 0.153 | 0.244 | 18 |
| 5 | DeBERTa PII (lakshyakh93) | 0.100 | 0.164 | 0.125 | 116 |

### Key Findings

Our model achieves an F1 score of **0.908**, representing a **272% improvement** over Piiranha (0.244), the best-performing existing model. Notably, Piiranha has high precision (0.600) but very low recall (0.153), meaning it misses the majority of PII entities — the exact failure mode identified by Singh & Narayanan. Our model maintains both high precision (0.898) and high recall (0.918), with overall accuracy of **97.9%** and ROC AUC of **0.9998**.

**Label Coverage:** Our model supports 505 entity labels compared to 18 for Piiranha, enabling fine-grained PII classification across names (first/middle/last), addresses (street/city/state/zip), financial identifiers (credit cards, bank accounts, tax IDs), and emerging PII types.

---

## Why Our Approach Works

Three factors contribute to our model's strong performance:

1. **Targeted synthetic data** — by generating samples specifically for NER failure modes, we provide training signal for edge cases that existing datasets lack
2. **DoRA fine-tuning** — weight decomposition enables efficient adaptation while preserving the pretrained model's linguistic knowledge
3. **Larger base model** — DeBERTaV3-large has superior contextual understanding compared to DistilBERT or base-sized models

---

## Limitations

- **Synthetic data bias:** Our multilingual dimension inadvertently included some non-English text that should have been filtered
- **Label fragmentation:** The 505-label schema may be overly granular for some applications
- **Computational cost:** DeBERTaV3-large requires significant GPU memory for inference compared to DistilBERT
- **Evaluation scope:** We evaluated on held-out test data from the same distribution; performance on truly out-of-distribution data (e.g., new PII formats) is untested

---

## Future Work

- Label consolidation to reduce the 505 labels to a more practical taxonomy
- Additional adversarial robustness testing
- Deployment optimization via quantization or distillation
- Extending to multilingual text (not just international PII formats in English)

---

## Resources

- **Model:** [huggingface.co/Ari-S-123/deberta-v3-large-pii-consolidated](https://huggingface.co/Ari-S-123/deberta-v3-large-pii-consolidated)
- **Dataset:** [huggingface.co/datasets/Ari-S-123/pii-detection-english-consolidated](https://huggingface.co/datasets/Ari-S-123/pii-detection-english-consolidated)
- **Code:** [github.com/Ari-S-123/pii-masking](https://github.com/Ari-S-123/pii-masking)

---

## References

1. Singh, D. & Narayanan, S. (2025). *Unmasking the Reality of PII Masking Models.* arXiv:2504.12308
2. Liu, S. et al. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML 2024.
3. He, P. et al. (2021). *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training.* arXiv:2111.09543
4. ai4privacy (2023). *Open PII Masking Dataset.* HuggingFace Datasets.

---

## Acknowledgments

This project would not have been possible to complete in the timeframe it was completed in without the help of Claude Opus 4.5 Thinking.
