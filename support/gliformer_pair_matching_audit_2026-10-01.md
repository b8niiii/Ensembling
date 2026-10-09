# GLiFormer directed-pair matching audit — reviewed validation 30

**Status:** completed offline on 1 October 2026. The audit read the frozen 30-bag validation inputs, confirmed text labels, and saved model outputs. Its machine-readable record is preserved under `pair_matching_audit` in the GLiFormer section of the [historical evidence snapshot](../results/preliminary_experiments.json), including hashes, rejected triples, the alternative rule, and cohort scores. No model was rerun and no official test was opened. Historical detailed outputs and the audit script were removed after verification; the original aggregate tables remain in [Preliminary_experiments.ipynb](../source/Preliminary_experiments.ipynb).

## Question and rule

The original mapping retains a GLiFormer triple only if its predicted head and tail names match the supplied **ordered** head → tail names after case-folding and whitespace normalization. It retained 8 of 27 raw triples; 19 were rejected. The proposed, separately named *location-qualifier* condition additionally accepts a triple only when:

1. The predicted head still matches the supplied head exactly under the original normalization.
2. The predicted tail is typed `location` by GLiFormer and begins with the exact supplied tail name followed by a comma and a nonempty qualifier, such as `San Jose , Calif.` for `San Jose`.
3. The original predicted relation is retained unchanged. The pair is never reversed, and no inverse relation is inferred.

This rule does not consult the reference label at inference time. It is an exploratory validation rule, not a frozen replacement for the original mapping. A comma-qualified name is not guaranteed to denote the same location in every case; further validation must check that risk.

## All 19 originally rejected raw triples

The target pair is in the reviewed manifest; each listed arrow is **the model's raw directed pair**. “Retain” means only the alternative location-qualifier rule would keep it. “Reject” means it is still ineligible for the supplied ordered pair; the underlying statement may nevertheless concern another real pair.

| Bag / raw triple | Predicted head → tail | Relation ID | Audit decision |
|---|---|---|---|
| C01 #1 | Port Moresby → Papua New Guinea | `/location/country/capital` | Reject: reversed country/capital pair. |
| C02 #1 | Jean-Fran çois Revel → French | `/people/deceasedperson/place_of_death` | Reject: another subject and tail. |
| C02 #2 | Jean-Fran çois Revel → Paris | `/people/deceasedperson/place_of_death` | Reject: target head is Voltaire. |
| C04 #1 | Brasília → Brazil | `/location/country/capital` | Reject: reversed country/capital pair. |
| C04 #2 | Brasília → Brazil | `/location/country/capital` | Reject: reversed country/capital pair. |
| C04 #3 | Brasília 18 % → Brazil | `/location/country/capital` | Reject: reversed pair with a noisy head span. |
| C05 #1 | Carlos Slim Domit → Carlos Slim Helú | `/people/person/children` | Reject: child/parent direction reversed. |
| C07 #1 | Mark Mays → Clear Channel | `/business/person/company` | Reject: another tail and relation for this target. |
| C07 #2 | Mark Mays → Lowry Mays | `/people/person/children` | Reject: child/parent direction reversed. |
| N02 #1 | Vincor → Mississauga | `/business/location` | Reject: another head; target is Canada → Mississauga. |
| N04 #1 | Jane Elizabeth Hodgson → Crookston , Minn. | `/people/person/place_of_birth` | **Retain:** exact head; location tail extends Crookston with a comma qualifier. |
| N07 #1 | Lloyd G. Trotter → G.E. | `/business/person/company` | Reject: another tail; target is African American. |
| N09 #1 | Sheryl Schwartz → Calvin Klein | `/business/person/company` | Reject: another head and reversed target orientation. |
| N09 #2 | Barry Schwartz → Calvin Klein | `/business/person/company` | Reject: reversed target pair; do not infer an inverse label. |
| N11 #1 | Theodor Uppman → San Jose , Calif. | `/people/person/place_of_birth` | **Retain:** exact head; location tail extends San Jose with a comma qualifier. |
| N13 #1 | Gustavo Cisneros → Cisneros Group | `/business/person/company` | Reject: another tail and reversed target orientation. |
| N13 #2 | Gustavo Cisneros → Univision | `/business/person/company` | Reject: reversed target pair; do not infer an inverse label. |
| N15 #1 | Barr → Woodcliff Lake , N.J. | `/business/location` | Reject: reversed target pair even though the location name has a qualifier. |
| N16 #1 | Silver Lake Partners → He | `/business/company/founders` | Reject: another tail; target is Seagate Technology. |

## Effect on the fixed reviewed reference

Only **N04 and N11** change. Both added place-of-birth relations agree with the separately confirmed text labels and were already predicted by **both** GLiNER-relex base and large. No false positive is added among these 30; that is an observed result on this sample, not a guarantee for other bags.

| Condition | New 18 TP / FP / FN; micro-F1 | Old 12 TP / FP / FN; micro-F1 | All 30 TP / FP / FN; micro-F1 |
|---|---:|---:|---:|
| GLiFormer original exact match | 4 / 0 / 8; 0.500 | 4 / 0 / 7; 0.533 | 8 / 0 / 15; 0.516 |
| GLiFormer location qualifier | 6 / 0 / 6; 0.667 | 4 / 0 / 7; 0.533 | 10 / 0 / 13; 0.606 |
| Base + large + GLiFormer, original 2-of-3 | 8 / 4 / 4; 0.667 | 7 / 1 / 4; 0.737 | 15 / 5 / 8; 0.698 |
| Base + large + GLiFormer, qualified 2-of-3 | 8 / 4 / 4; 0.667 | 7 / 1 / 4; 0.737 | 15 / 5 / 8; 0.698 |
| Relex large alone | 12 / 11 / 0; 0.686 | 10 / 6 / 1; 0.741 | 22 / 17 / 1; 0.710 |

The individual GLiFormer F1 improves by **0.090** on all 30, but the three-member vote does **not change on any bag**: base and large already agreed on the two recovered labels. Under this narrow alternative, there is still no measured gain over relex large alone to justify a third call. The 30 bags were selected for diagnosis, and the older 12 were used in earlier screening; neither result estimates generalization. Further GLiFormer work is deferred; any reopening requires a new broader validation of the mapping and negative-bag errors before freezing it. Model-training overlap and controlled end-to-end cost remain separate open checks.
