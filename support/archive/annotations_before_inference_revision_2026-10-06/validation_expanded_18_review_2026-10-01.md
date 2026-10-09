# Confirmed 18-bag validation review

**Status:** all 18 proposed text-based annotations were confirmed without changes on 1 October 2026, including N06 and N15. The confirmed machine-readable reference is in [validation_expanded_18_annotations_2026-10-01.json](validation_expanded_18_annotations_2026-10-01.json). These are development annotations, not official test gold. The four-encoder [reviewed-30 run](../../../source/Preliminary_experiments.ipynb) and separate [GLiFormer Large screen](../../../source/Preliminary_experiments.ipynb) subsequently completed on these 18 plus the earlier 12.

**Annotation rule:** read each sentence for the ordered head → tail pair. Keep a relation only if the supplied sentence expresses it; otherwise use `[]`. The input-only sample deliberately contains no answer labels; the confirmed reference is stored separately. The reviewed boundary cases N01, N06, N09, N10, N15, N17 and N18 retain their proposed labels.

**Sampling:** 18 distinct one-sentence NYT10m validation bags; neither their pairs nor their sentence texts appeared in the earlier pilot/extended manifests or prior reviewed samples. No ordered pair or punctuation-insensitive sentence matches NYT10m train. The published NYT11 train was unavailable for this pass, so this sample is **not yet cleared for evaluating GenTune**. Distant labels and model predictions are withheld here. This is a purposeful diagnostic sample, not a random estimate of model performance.

## N01 — Montserrat → Plymouth

Pair IDs: `m.04wcf → m.05vw7`.

> The eruptions demolished much of Montserrat , a British colony , including its capital , Plymouth , and forced the evacuation of two-thirds of the island 's inhabitants .

**Proposed relations:** `/location/country/capital`, `/location/location/contains`.

**Reason to review:** The sentence calls Plymouth the capital of Montserrat; review whether the capital→containment convention applies.

- [x] Confirmed without changes.

## N02 — Canada → Mississauga

Pair IDs: `m.0d060g → m.0154gx`.

> Vincor is based in Mississauga , Canada , which is near Toronto .

**Proposed relations:** `/location/location/contains`.

**Reason to review:** “Mississauga, Canada” places the city in the country.

- [x] Confirmed without changes.

## N03 — Mike Jeffries → Abercrombie & Fitch

Pair IDs: `m.026tkbc → m.02z2m_`.

> Mike Jeffries , the chief executive of Abercrombie & Fitch , does n't want you in his clothing stores . ''

**Proposed relations:** `/business/person/company`.

**Reason to review:** Jeffries is explicitly called the company’s chief executive.

- [x] Confirmed without changes.

## N04 — Jane Elizabeth Hodgson → Crookston

Pair IDs: `m.026g4n0 → m.0wdjl`.

> Jane Elizabeth Hodgson was born on Jan. 23 , 1915 , in Crookston , Minn. .

**Proposed relations:** `/people/person/place_of_birth`.

**Reason to review:** The sentence says she was born in Crookston.

- [x] Confirmed without changes.

## N05 — Hrishikesh Mukherjee → Mumbai

Pair IDs: `m.0674cw → m.04vmp`.

> Hrishikesh Mukherjee , who produced and directed memorable and successful Hindi films in a career of more than five decades , died on Aug. 27 in Mumbai .

**Proposed relations:** `/people/deceasedperson/place_of_death`.

**Reason to review:** The sentence says he died in Mumbai.

- [x] Confirmed without changes.

## N06 — Lloyd Maines → Natalie Maines

Pair IDs: `m.02l8jr → m.018pyg`.

> And his recent album , '' Good Times '' -LRB- Dualtone -RRB- , was produced by Lloyd Maines , father of Natalie Maines -- the third Chick .

**Proposed relations:** `/people/person/children`.

**Reason to review:** “Lloyd Maines, father of Natalie Maines” explicitly states head = parent and tail = child. `/people/person/children` names the children of the head; there is no separate fatherhood label. A [targeted aggregate test-label audit](../../../results/preliminary_experiments.json) found 54 test bags with this label, including nine unordered pairs labeled in both directions. Thus the test is not perfectly directional, but that inconsistency does not reverse the evidence in this sentence. The label was subsequently confirmed on 1 October 2026.

- [x] Confirmed without changes.

## N07 — Lloyd G. Trotter → African American

Pair IDs: `m.03wpv91 → m.0x67`.

> Lloyd G. Trotter , an African American who has worked for G.E. for 35 years and is now chief executive of GE Consumer and Industrial , a $ 13 billion business , doubts that racism results in white employees getting jobs that their black colleagues covet . ''

**Proposed relations:** `/people/person/ethnicity`.

**Reason to review:** Trotter is explicitly described as African American.

- [x] Confirmed without changes.

## N08 — Krispy Kreme → Winston-Salem

Pair IDs: `m.015vw5 → m.0ygbf`.

> Mr. Jervik , 49 , will receive at least $ 400,000 in pay next year , but there is no sugar coating on one clause in his contract : he must relocate to be near Krispy Kreme 's headquarters in Winston-Salem , N.C. Patrick McGeehan OPENERS : SUITS

**Proposed relations:** `/business/location`.

**Reason to review:** The company’s headquarters are explicitly located in Winston-Salem.

- [x] Confirmed without changes.

## N09 — Calvin Klein → Barry Schwartz

Pair IDs: `m.027m0s → m.08wxz6`.

> There is a hole in everyone 's heart , '' said Sheryl Schwartz , the wife of Barry Schwartz , a co-founder of Calvin Klein and a prominent thoroughbred owner and breeder . ''

**Proposed relations:** `/business/company/founders`.

**Reason to review:** Barry Schwartz is called a co-founder of Calvin Klein; review the company reading of “Calvin Klein.”

- [x] Confirmed without changes.

## N10 — Lou Ye → China

Pair IDs: `m.02plttk → m.0d05w3`.

> China Punishes Filmmaker The Chinese director Lou Ye has been banned from filmmaking for five years because he submitted his '' Summer Palace '' to the Cannes Film Festival without official approval , Reuters reported .

**Proposed relations:** `/people/person/nationality`.

**Reason to review:** “Chinese director Lou Ye” supports nationality; review whether “Chinese” is sufficiently explicit under our rule.

- [x] Confirmed without changes.

## N11 — Theodor Uppman → San Jose

Pair IDs: `m.05lsgj → m.0f04v`.

> Theodor Uppman was born in San Jose , Calif. , on Jan. 12 , 1920 , to a family of Swedish heritage .

**Proposed relations:** `/people/person/place_of_birth`.

**Reason to review:** The sentence says Uppman was born in San Jose; it does not itself establish later residence.

- [x] Confirmed without changes.

## N12 — James Manning → Providence

Pair IDs: `m.052f_2 → m.0c1d0`.

> The university 's founder , the Rev. James Manning , freed his only slave , but accepted donations from slave owners and traders , including the Brown family of Providence , R.I. .

**Proposed relations:** `[]` (no allowed relation explicitly stated).

**Reason to review:** Providence describes the Brown family, not James Manning’s place of death.

- [x] Confirmed without changes.

## N13 — Univision → Gustavo Cisneros

Pair IDs: `m.02p10m → m.048yyh`.

> Gustavo Cisneros , chairman of the Cisneros Group , has been a member of Univision 's board since 2003 .

**Proposed relations:** `[]` (no allowed relation explicitly stated).

**Reason to review:** Board membership does not state that Cisneros is a major shareholder of Univision.

- [x] Confirmed without changes.

## N14 — Dyson → James Dyson

Pair IDs: `m.05cskg → m.0209vm`.

> Best foreign film -- James Dyson , the British inventor of the Dyson vacuum cleaner , gave a stirring performance in a commercial for his product line .

**Proposed relations:** `[]` (no allowed relation explicitly stated).

**Reason to review:** Inventing a Dyson product does not explicitly state that James Dyson founded the company.

- [x] Confirmed without changes.

## N15 — Woodcliff Lake → Barr

Pair IDs: `m.02079bk → m.04b2m_7`.

> Based in Woodcliff Lake , N.J. , Barr specializes in contraceptives and generic drugs .

**Proposed relations:** `[]` (no allowed relation explicitly stated).

**Reason to review:** The sentence places the company Barr in Woodcliff Lake, but the query is location → company. `/business/location` would run in the reverse direction, and `/location/location/contains` ordinarily takes a location as its tail. In the [targeted aggregate test-label audit](../../../results/preliminary_experiments.json), reverse pairs of test `/business/location` bags are inconsistently labeled `contains` (9), `NA` (7), both (4), or absent (109); neither N15 entity occurs in test. This does not establish a stable test convention for a corporate headquarters. `[]` was subsequently confirmed on 1 October 2026 under the strict directed ontology.

- [x] Confirmed without changes.

## N16 — Silver Lake Partners → Seagate Technology

Pair IDs: `m.02qcy82 → m.01wdsm`.

> He co-founded Silver Lake Partners , the first private equity firm to focus successfully on technology companies , buying up businesses like Seagate Technology .

**Proposed relations:** `[]` (no allowed relation explicitly stated).

**Reason to review:** The sentence mentions Seagate among purchases but states no relation in our ontology for this ordered pair.

- [x] Confirmed without changes.

## N17 — London → Woolwich

Pair IDs: `m.04jpl → m.0nczk`.

> It went into service in 1982 at Woolwich , about 10 miles east of central London .

**Proposed relations:** `[]` (no allowed relation explicitly stated).

**Reason to review:** “East of central London” gives a relative position but does not explicitly say that London contains Woolwich; review this boundary case.

- [x] Confirmed without changes.

## N18 — Andrea Barzagli → Italy

Pair IDs: `m.09bvds → m.03rjj`.

> So Italy will use 25-year-old Andrea Barzagli in Nesta 's spot .

**Proposed relations:** `[]` (no allowed relation explicitly stated).

**Reason to review:** Playing for Italy does not explicitly establish Barzagli’s nationality; review this strict-evidence judgment.

- [x] Confirmed without changes.
