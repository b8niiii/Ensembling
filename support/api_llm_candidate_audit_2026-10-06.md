# Open-weight API LLM candidate and budget audit

Verified 6 October 2026. Public model cards, provider documentation and API metadata were read. No inference API was called, no project data was uploaded, and no checkpoint weights were downloaded. Candidate selection and the final API configuration remain open.

## Scope and benchmark evidence

Candidates cover Xiaomi, Z.ai, Alibaba, DeepSeek, Mistral and NVIDIA. Public weight releases are linked below. Qwen and NVIDIA use their stated custom licenses; open weights do not imply identical licensing or a free hosted API. API metadata links MiMo, Qwen, Mistral and NVIDIA endpoints to the listed releases; the DeepSeek documentation explicitly identifies its served version. This identifies the advertised release, not byte-for-byte serving implementation or provider quantization.

The [Artificial Analysis leaderboard](https://artificialanalysis.ai/leaderboards/models), Intelligence Index v4.3.2, supplies a common general-capability comparison. Scores below refer to the reasoning configurations displayed on that leaderboard, including max effort for DeepSeek. They are not NYT10m F1 or an RE benchmark, and cannot be attributed to a non-thinking configuration. Several component evaluations emphasize agents, coding or difficult reasoning. No task winner is established by this audit.

## Full-validation cost assumptions

The unchanged supplied-entity manifest contains 36,266 bags and 44,751 selected sentence records, with the existing deterministic cap of 20. Budget inputs include one shared English instruction/24-relation-description prefix and JSON head, tail, sentence text and exact spans, without reference labels, demonstrations or tools. One request is assumed per bag. This is a draft prompt for sizing, not a frozen experimental prompt.

- Manifest SHA256: `7401a243244e6a8972d2d5584a1fe308f638d1412f151ef5f24c77a0b69fd6f6`.
- Draft system prompt SHA256: `70da3990f6e6a0ba34646f2fd6e21a11c77a79a50564eadce4f53d5448536924`; 2,006 characters.
- All draft requests combined: 87,100,665 input characters.
- Local tokenizer counts, with an assumed 24-token chat overhead per request: MiMo Pro 18,837,322 tokens (519.42/bag); GLM Flash 18,555,219 (511.64/bag). API serialization, exact chat templates and other tokenizers can change the billed inventory.
- A common rounded **20,000,000 input tokens** is used for the comparable budget table; it is not a tokenizer-specific measurement for every candidate.
- Final-output scenario: **80 tokens per bag**, totaling 2,901,280 tokens. This is a budget assumption, not observed generation or a proposed hard output cap.
- Reasoning scenario: the same JSON plus **1,000 reasoning tokens per bag**, totaling 39,167,280 output tokens. This is a sensitivity assumption, not a measured average or a guaranteed reasoning-token limit.
- USD, uncached input, one pass, no retries, no discounts, no provider top-up/payment fees or applicable tax. Add a contingency after the actual pilot usage is available.

`cost = input_tokens / 1e6 * input_rate + billed_output_tokens / 1e6 * output_rate`

| Model / public weights | Origin | General index | License | Input / output USD per 1M | 80 final tokens, no reasoning | Plus 1,000 reasoning tokens |
|---|---|---:|---|---:|---:|---:|
| [MiMo-V2.6-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) | China | 38 | MIT | [0.14 / 0.28](https://mimo.mi.com/docs/pricing) | $3.61 | $13.77 |
| [MiMo-V2.6-Pro](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) | China | 46 | MIT | [0.435 / 0.87](https://mimo.mi.com/docs/pricing) | $11.22 | $42.78 |
| [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | China | 42 | MIT | [0.15 / 0.5](https://docs.z.ai/guides/overview/pricing) | Unavailable (reasoning required) | $22.58 |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | China | 40 | Qwen Community 1.0 | [0.15 / 0.47](https://openrouter.ai/qwen/qwen3.8-flash) | $4.36 | $21.41 |
| [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | China | 39 | MIT | [0.15 / 0.6](https://api-docs.deepseek.com/quick_start/pricing/) | $4.74 | $26.50 |
| [Mistral Small 4](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) | France | 11 | Apache 2.0 | [0.15 / 0.6](https://docs.mistral.ai/inference/pricing) | $4.74 | $26.50 |
| [Nemotron 3 Ultra](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16) | United States | 23 | OpenMDW 1.1 | [0.5 / 2.2](https://openrouter.ai/nvidia/nemotron-3-ultra-550b-a55b) | $16.38 | $96.17 |

DeepSeek rates are **off-peak**. Peak prices and the corresponding totals are double: $0.30/$1.20 per 1M, approximately $9.48 without reasoning or $53.00 in the 1,000-token reasoning scenario. The provider defines peak hours as 01:00–04:00 and 06:00–10:00 UTC on weekdays, excluding Chinese public holidays. The current API model ID is `deepseek-flash`; older names may be routed to V4.1.

MiMo offers a documented **50% batch discount**: Flash approximately $1.81/$6.88 and Pro $5.61/$21.39 for the two scenarios. These discounted totals are distinct from the standard table. Prompt-cache savings are excluded because cache eligibility, minimum prefix length and actual hits have not been measured.

## Reasoning and next decision

- [GLM-5.3-Flash documentation](https://docs.z.ai/guides/vlm/glm-5.3-flash) states that thinking cannot be disabled. Its $4.45 arithmetic no-reasoning lower bound is not an available inference condition.
- [MiMo API documentation](https://mimo.mi.com/docs/en-US/api/chat/anthropic-api) supports enabled/disabled thinking. Its Responses API maps `reasoning.effort=none` to non-thinking, while low/medium/high do not establish distinct supported token budgets.
- [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) documents `enable_thinking=False`; [DeepSeek pricing/model documentation](https://api-docs.deepseek.com/quick_start/pricing/) lists both modes. Mistral and NVIDIA model cards also describe configurable reasoning. Hosted endpoint controls must be verified when implementing the selected provider.

MiMo Pro is a candidate for the strong large-model comparator; MiMo Flash is the lowest-priced candidate in this shortlist; GLM Flash and Qwen Flash-Next provide alternative capability/cost points. Mistral Small 4 and Nemotron Ultra provide non-Chinese open-weight alternatives but have lower scores on this particular common index. No published result found here establishes their comparative NYT10m performance.

A bounded setup pilot can reuse the reviewed 30 for errors and the existing fixed random100 for broader behavior and usage measurement. Record billed input, final-output and reasoning tokens, invalid/truncated responses, retries and latency. Compare thinking and non-thinking as declared development configurations where supported. Freeze the model/version, provider, prompt and decoding configuration before the official test. No additional annotation or full-validation API run is scheduled by this audit.

Reasoning dominates cost quickly: MiMo Pro at an assumed 3,000 reasoning tokens/bag would cost about $105.88 before discounts or retries, rather than $42.78. A short JSON response does not guarantee a short paid reasoning trace.

## Tokenizer provenance

- MiMo Pro tokenizer-only revision: `73875d00b30a89ef8cc353a0b60b0e9f9561952d`.
- GLM Flash tokenizer-only revision: `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`.
- Only tokenizer JSON files were downloaded to the temporary directory. Token counting remained local and did not load models.

## Draft shared system prompt used for sizing

```text
Extract relations from the supplied news evidence for the ordered HEAD -> TAIL pair. Return exactly one JSON object with the key "relations" and a list of distinct allowed relation IDs. Return {"relations": []} when no allowed relation is supported. Use the supplied mentions; do not detect new entities. Preserve relation direction. Multiple supported relations may be returned. Judge from the sentences; distinguish supported linguistic inference from unsupported outside knowledge. Output no explanations, Markdown or additional keys. Allowed relations (head-to-tail meaning):
/business/company/advisors: company has advisor
/business/company/founders: company has founder
/business/company/majorshareholders: company has major shareholder
/business/company/place_founded: company was founded in place
/business/location: business is located in place
/business/person/company: person is associated with company
/film/film/featured_film_locations: film features location
/location/administrative_division/country: administrative division belongs to country
/location/country/administrative_divisions: country has administrative division
/location/country/capital: country has capital city
/location/location/contains: location contains location
/location/neighborhood/neighborhood_of: neighborhood belongs to location
/location/region/capital: region has capital city
/location/us_county/county_seat: US county has county seat
/people/deceasedperson/place_of_burial: deceased person was buried in place
/people/deceasedperson/place_of_death: deceased person died in place
/people/ethnicity/geographic_distribution: ethnicity is distributed in location
/people/person/children: person has child
/people/person/ethnicity: person belongs to ethnicity
/people/person/nationality: person has nationality
/people/person/place_lived: person lived in place
/people/person/place_of_birth: person was born in place
/people/person/religion: person follows religion
/time/event/locations: event occurred in location
```
