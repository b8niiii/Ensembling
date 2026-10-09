# Overlap-free validation sample: confirmed text annotations

**Status:** Human-confirmed on 29 September 2026, before the specialist screen. These are text-only development annotations, not official test gold labels.

**Subsequent use:** these 12 bags form the older cohort of the completed [30-bag encoder comparison](../source/Preliminary_experiments.ipynb) and the separate completed [GLiFormer Large screen](../source/Preliminary_experiments.ipynb). The label sets remain unchanged; the current model decision also requires the new-18 cohort and broader development evidence.

**Selection:** Twelve complete NYT10m validation bags (17 sentences total; maximum four per bag). No selected sentence matches published NYT11 train by exact text, normalized text, or punctuation-insensitive word sequence. Distant labels were used only to balance sampling strata (four `NA`, six single-positive, two multi-positive); they were withheld from the annotation input. The sample is diagnostic, not prevalence-representative.

**Review update:** C01/C04 retain both country–capital and broad containment. In the distributed test labels, all 18 country–capital bags also carry containment; this aggregate-only test-label check is disclosed in the protocol. C03 remains `[]`: West Side is the location, while Wal-Mart is the company, so neither the location→location containment label nor the business→location label fits the ordered pair. C12 retains company→founder, the direction used by the ontology and training examples. C07 retains the explicit father→son relation despite its distant `NA` stratum. These decisions were frozen before model screening.

## C01 — Papua New Guinea → Port Moresby

Pair IDs: `m.05qkp → m.0fs0v`.

1. Port Moresby , the capital of Papua New Guinea , has for decades been portrayed in the Western press as a lawless place overrun by ferocious gangs of '' raskols . ''

**Confirmed relations:** `/location/country/capital`, `/location/location/contains`.
**Evidence:** sentence 1. Port Moresby is explicitly called the capital of Papua New Guinea. The broader containment label follows from the stated capital-of relation and matches the test-label co-occurrence convention.

## C02 — Voltaire → Paris

Pair IDs: `m.07ym0 → m.05qtj`.

1. Abdellatif Kechiche 's first film , about a young North African immigrant 's adventures in Paris , was called '' It 's Voltaire 's Fault . ''
2. During the seasonal collections in Europe , they eat at the same three or four restaurants in Milan -LRB- da Giacomo -RRB- and Paris -LRB- Voltaire or Costes -RRB- .
3. Jean-Fran çois Revel , a prolific philosopher , writer and journalist who summoned the classical polemical weapons of Voltaire and Montaigne , including humor , irony and surprise , to illuminate subjects from French cuisine to French anti-Americanism , died on Saturday in Paris .

**Confirmed relations:** none (`[]`).
**Evidence:** no positive relation stated. The apparent Voltaire mentions refer to a film title, a Paris restaurant name, and a comparison with another writer. No relation between Voltaire the person and Paris is stated.

## C03 — West Side → Wal-Mart

Pair IDs: `m.02w9slf → m.0841v`.

1. In the coming weeks , Wal-Mart expects to open its first store in Chicago , on the city 's West Side .

**Confirmed relations:** none (`[]`).
**Evidence:** no positive relation stated. The sentence locates a planned Wal-Mart store on the West Side. The available company-to-location relation has the opposite direction from this query.

## C04 — Brazil → Brasília

Pair IDs: `m.015fr → m.01hy_`.

1. We started because it was something the business community asked for and to deter the mafia falsifying visas around our consulates , '' the Mexican ambassador to Brazil , Cecília Soto , said in a telephone interview from Brasília . ''
2. What 's on the table today , taking into account all aspects , is n't sufficient , '' Celso Amorim , Brazil 's foreign minister , said at a news conference in Brasília . ''
3. On Friday , lawyers representing the husband of a crash victim asked a court in Brasília , the capital , to order that the Legacy remain in Brazil for now .
4. In '' Brasília 18 % , '' a distinguished medical examiner , summoned to Brazil 's administrative capital to confirm the identity of a beautiful congressional aide found dead , finds himself a pawn in a political conspiracy that extends to the highest levels of government .

**Confirmed relations:** `/location/country/capital`, `/location/location/contains`.
**Evidence:** sentence 3, sentence 4. Brasília is called the capital in sentence 3 and Brazil’s administrative capital in sentence 4. The broader containment label follows from this stated country–capital relationship and matches the test-label co-occurrence convention.

## C05 — Carlos Slim Helú → Carlos Slim Domit

Pair IDs: `m.02ygr1 → m.02vsjy0`.

1. LAST week , Carlos Slim Domit , the eldest son of Carlos Slim Helú , the Mexican billionaire , confirmed what had earlier been largely dismissed as a rumor : he had acquired a stake in Univision , the big Spanish-language television and radio company that recently put itself on the block .

**Confirmed relations:** `/people/person/children`.
**Evidence:** sentence 1. Carlos Slim Domit is explicitly called the eldest son of Carlos Slim Helú.

## C06 — Evo Morales → Aymara

Pair IDs: `m.01pt2r → m.01g3rx`.

1. To judge by the overwhelming victory of Evo Morales , an Aymara , in Bolivia 's elections on Dec. 18 , he kept his promise .

**Confirmed relations:** `/people/person/ethnicity`.
**Evidence:** sentence 1. Evo Morales is explicitly described as an Aymara.

## C07 — Lowry Mays → Mark Mays

Pair IDs: `m.03j0v_ → m.0930r6`.

1. Mark Mays , Clear Channel 's chief executive and the son of Lowry Mays , the company 's founder , said on a conference call with analysts yesterday that the company wanted to allow '' our business units to grow as fast as they possibly can , and as we have looked at different ways to accomplish that over the last couple of years , we believe that this absolutely does that . ''

**Confirmed relations:** `/people/person/children`.
**Evidence:** sentence 1. Mark Mays is explicitly called a son of Lowry Mays, despite this bag’s distant-supervision NA stratum.

## C08 — Matt Damon → Leonardo DiCaprio

Pair IDs: `m.0169dl → m.0dvmd`.

1. Based on the crackling Hong Kong thriller '' Infernal Affairs , '' it features fine twinned performances from Matt Damon and Leonardo DiCaprio , and a showboating Jack Nicholson .

**Confirmed relations:** none (`[]`).
**Evidence:** no positive relation stated. The two actors are mentioned as performers in the same film, which is not an allowed relation for the ordered pair.

## C09 — Ismail Merchant → Mumbai

Pair IDs: `m.03_80b → m.04vmp`.

1. Ismail Merchant Buried At a ceremony attended by relatives and friends , Ismail Merchant , the producer of award-winning films like '' Howards End , '' '' A Room With a View '' and '' The Remains of the Day , '' was buried on Saturday in Mumbai , formerly Bombay , where he was born in 1936 , the BBC reported .

**Confirmed relations:** `/people/deceasedperson/place_of_burial`, `/people/person/place_of_birth`.
**Evidence:** sentence 1. The sentence says Ismail Merchant was buried in Mumbai and was born there.

## C10 — Russia → Saudi Arabia

Pair IDs: `m.06bnz → m.01z215`.

1. Cuba , China , Russia , Pakistan and Saudi Arabia , on the other hand , sought and won seats .

**Confirmed relations:** none (`[]`).
**Evidence:** no positive relation stated. Russia and Saudi Arabia are listed together as countries that won seats. No allowed direct relation is stated.

## C11 — Mo Mowlam → Canterbury

Pair IDs: `m.01qhsj → m.0cy07`.

1. Mo Mowlam , the earthy , straight-talking politician who as Britain 's Northern Ireland secretary helped negotiate the Good Friday peace settlement in 1998 , died Friday morning at a hospice in Canterbury , her family said .

**Confirmed relations:** `/people/deceasedperson/place_of_death`.
**Evidence:** sentence 1. Mo Mowlam is explicitly said to have died at a hospice in Canterbury.

## C12 — Facebook → Mark Zuckerberg

Pair IDs: `m.02y1vz → m.086dny`.

1. YAHOO BIDS FOR FACEBOOK -- To woo Mark Zuckerberg , the 22-year-old founder of the social networking Web site Facebook , Yahoo has offered about $ 900 million for the company and said it would keep Mr. Zuckerberg in charge .

**Confirmed relations:** `/business/company/founders`.
**Evidence:** sentence 1. Mark Zuckerberg is explicitly called the founder of Facebook, matching the company-to-founder direction.

---

Source validation SHA-256: `614aa5be9dd678d146cf8d44029e4c993f82add560ef7c00dfcb366416bed18c`. NYT11 train Parquet SHA-256: `45cdd73a004157aa42994a1a3f7f25de727a7841d0a529e6032bace4148ab15c`. Frozen sample SHA-256: `00dd4aa10878b6aed80134cc575b88879bd45bf35f2e32fd71b06cc113f23b9e`.

**Policy revision, 6 October 2026:** documented motivated inference is explicitly accepted alongside direct text support. No C01–C12 labels changed.
