# Customer Feedback Intelligence | AI Quality Assurance Mini-Project

**AI Literacy · Customer Experience · Business Analysis · Critical Thinking**

## Project overview

This 90-minute portfolio exercise explores how generative AI can help a fictional café in Adelaide, South Australia, understand customer feedback—and why its analysis still needs verification.

> **Data transparency:** All 30 reviews are synthetic, AI-generated examples. No real customers or café were studied. The work demonstrates a method, not claims about a real business.

**Workflow:** Generate 30 fictional reviews → analyse with NotebookLM → challenge the analysis with a second QA prompt → independently review discrepancies → document defensible findings and limitations.

## Tools and files

- **ChatGPT/another generative AI:** generated the initial synthetic review dataset using a structured prompt.
- **Google NotebookLM:** produced the initial sentiment and business analysis, then a QA re-analysis.
- **Independent review:** checked source review IDs and assessed whether issue counts were supported.

| File | Purpose |
| --- | --- |
| `customer_reviews.csv` | The 30 synthetic source reviews (IDs R001–R030) |
| `prompts.md` | Dataset creation and analysis/audit prompts |
| `validation.md` | Audit trail, verified issue definitions and remaining caveats |

## 1. Initial AI-generated analysis

NotebookLM reviewed all 30 reviews and returned the following **initial** labels:

| Sentiment | Initial count |
| --- | ---: |
| Positive | 10 |
| Negative | 10 |
| Mixed | 8 |
| Neutral | 2 |
| **Total** | **30** |

It also reported three equally frequent issue clusters (7 mentions each). The categories bundled unrelated issues and sometimes included positive experiences as examples of negative failures. These totals should not be treated as verified.

## 2. Follow-up AI audit

A second NotebookLM pass reclassified **R024** from *Neutral* to *Mixed*. R024 explicitly praised the coffee while criticising the cakes and noise; under the stated rule, it is a mixed review.

| Sentiment | After QA review |
| --- | ---: |
| Positive | 10 |
| Negative | 10 |
| Mixed | 9 |
| Neutral | 1 |
| **Total** | **30** |

These are **AI-produced labels reviewed for plausibility**, not an independently annotated 30-row ground-truth benchmark. A prior 10-review spot check found nine agreements and one dispute (R024), which was resolved by clarifying the rubric; this is **agreement on a sample**, not a validated model-accuracy score.

## 3. Independent issue verification

To avoid overlapping category confusion, I used transparent definitions. **Counts are numbers of unique reviews within each category**, not numbers of individual incidents. **Categories can overlap**, so do not add counts together to get 30.

| Negative issue | Reviews | IDs |
| --- | ---: | --- |
| Food/drink quality or serving-temperature complaints | **8** | R001, R007, R009, R013, R022, R024, R025, R028 |
| Order errors or missed dietary instructions | **5** | R002, R010, R012, R018, R027 |
| Wait time or slow acknowledgement/service complaints | **5** | R001, R010, R016, R018, R025 |
| Price/value complaints | **5** | R007, R015, R021, R025, R028 |
| Cleanliness/hygiene complaints | **2** | R005, R019 |
| Dietary option stockout | **1** | R030 |

**Boundary rules:** Hair found in food (R019) is counted here as hygiene, not an order error. Dirty cups/tables (R005) are cleanliness, not food preparation quality. Wrong drink temperature against a specific request (R027) is an order-instruction failure rather than a general product-quality trend. The positively managed table wait (R023) is **not** counted as a negative service-delay complaint.

These counts describe this **small synthetic sample** only. They are not population estimates, and frequency alone does not establish severity or root cause. For example, dietary failures may merit higher priority than more frequently mentioned low-impact complaints.

## 4. Corrections and lessons learned

| Finding | Correction / interpretation |
| --- | --- |
| R024 called Neutral | Reclassified as Mixed after establishing an explicit rubric |
| Three issue groups each reported as 7 | Recounted with transparent, non-exclusive issue definitions and exact supporting IDs |
| Positive R023 wait included in negative wait problems | Excluded from negative complaints; treated as positive service recovery |
| Possible changes to beans/baristas inferred from R022 | Customer speculation, **not** proof of an operational change |
| “Churn risk” asserted from neutral reviews | A hypothesis requiring actual retention behaviour, not a fact from the reviews |
| Second audit labelled R005 a product-quality defect and R019 an order error | Corrected to hygiene/cleanliness categories |
| Second audit claimed 100% attribution accuracy | **Not independently established** for all source claims; do not repeat as verified |

The two AI passes illustrate an important limitation: **asking an AI to audit itself can fix some errors and introduce or retain others**.

## 5. Business recommendations

1. **Order and dietary safety:** Use a simple ticket verification step for milk substitutions, vegan requests and order modifiers. Treat dietary safety as high impact, not just a frequency metric (R002, R012, R027).
2. **Peak-period operations:** Track ticket times, acknowledge guests promptly and provide clear wait updates during busy periods (R001, R010, R016, R018, R025).
3. **Quality and value standards:** Review drink calibration, serving temperature, food consistency and perceived value; then monitor the same complaint categories after changes (R007, R009, R015, R022, R025, R028).

These are **proposed interventions**, not proven fixes. A real café would need operational data, staff consultation and follow-up measurement before implementing them.

## 6. Method and limitations

- Dataset: 30 synthetic English-language reviews with star ratings; intentionally includes praise, complaints and ambiguous cases.
- Sentiment labels: an explicit rubric distinguishes Neutral (*no distinct positive and negative claims*) from Mixed (*both clear praise and criticism*).
- QA: source-based spot review and re-counting of negative issue IDs.
- **No API-level model parameters were controlled**, and no measured performance benchmark across AI products was conducted.
- No real customer information, private records, recordings or confidential company data were used.
- This is a learning exercise in grounded analysis, not a scientific validation study.

## Key takeaway

> AI can speed up the first pass of customer feedback analysis, but reliable business insight depends on clearly defined categories, source-level evidence and human judgement.

## Learning context

Created as a practical portfolio exercise inspired by **Alura Latam — AI Literacy: Essential Skills and Critical Thinking**, especially the modules on generative AI use cases, analytical validation, hallucinations and privacy.
