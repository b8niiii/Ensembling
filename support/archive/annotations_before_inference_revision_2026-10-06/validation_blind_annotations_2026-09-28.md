# Annotazioni proposte su 12 bag di validation

Queste annotazioni sono state formulate leggendo le frasi e successivamente riesaminate. Sono un riferimento di lavoro per questo piccolo campione, **non le annotazioni ufficiali del test**. Il campione è stato fissato prima della lettura: tre bag per ciascuno dei quattro gruppi del pilota, poi mescolate; i tre esempi già mostrati in `SLM_pilot.ipynb` sono stati esclusi. Le predizioni dei modelli e le etichette di distant supervision delle 12 bag non sono state consultate durante l'annotazione. [Campione con tutte le 86 frasi](../../validation_blind_sample_2026-09-28.json) · [Etichette in formato JSON](validation_blind_annotations_2026-09-28.json).

**Stato successivo:** questo è il campione storico di audit dei primi decoder; la selezione attuale dei modelli usa ulteriori bag di validation revisionate, riassunte in [notes.md](notes.md). Le annotazioni qui sotto rimangono invariate.

La regola è usare solo le frasi fornite, le 24 relazioni ammesse e la direzione **head → tail**. Una semplice compresenza non basta; `[]` significa che il testo non sostiene alcuna relazione ammessa. Per le località, una forma come “Helsinki, Finland” sostiene il contenimento geografico, ma non basta a dimostrare che Helsinki sia la capitale.

Legenda: **contiene** = `/location/location/contains`; **persona–azienda** = `/business/person/company`; **nessuna** = `{"relations": []}`.

| Bag | Coppia ordinata | Risposta proposta | Indizio decisivo nelle frasi | Certezza |
| --- | --- | --- | --- | --- |
| B01 | Atlanta → Turner Field | contiene | «In Atlanta … outside … Turner Field» (frase 1) colloca lo stadio nell'area di Atlanta. | Media |
| B02 | Honolulu → Hawaii Kai | nessuna | «Hawaii Kai, **near** Honolulu» (1): *vicino* non prova *dentro*. | Alta |
| B03 | NBC → American Express | nessuna | Un programma NBC menziona una carta American Express (1); nessuna relazione diretta ammessa è dichiarata. | Alta |
| B04 | Jeffrey Katzenberg → DreamWorks Animation | persona–azienda | È chiamato «the chief executive of DreamWorks Animation» (1–3). | Alta |
| B05 | Queens → Astoria | contiene | «Astoria, Queens» compare ripetutamente; la frase 14 colloca un'attività di Queens ad Astoria. | Alta |
| B06 | Betty Smith → Brooklyn | nessuna | Brooklyn è nel titolo di *A Tree Grows in Brooklyn*, scritto da Smith (1–4); il testo non dice che lei vi abitò o vi nacque. | Alta |
| B07 | Finland → Helsinki | contiene | «Helsinki, Finland» (1–3) colloca la città nel Paese; le frasi non la chiamano capitale. | Alta |
| B08 | Iran → Tehran | nessuna | «Iranian government … in Tehran» (5) e altri usi di Tehran per il governo iraniano suggeriscono un legame, ma non dichiarano esplicitamente né contenimento né capitale. | Media: confermata la lettura stretta |
| B09 | New York → Queens | contiene | Il testo parla dei «New York neighborhoods» e cita Jamaica in Queens (12); elenca Queens tra le aree di New York (14). | Alta |
| B10 | Sam → Jon | nessuna | L'annuncio funebre menziona Sam come figlio e Jon come nipote del defunto (1), senza stabilire una relazione genitore–figlio **fra Sam e Jon**. | Alta |
| B11 | Scotland → Edinburgh | contiene | «Edinburgh, Scotland» (1–4) colloca la città in Scozia; il testo non la definisce capitale. | Alta |
| B12 | DeKalb County → Derwin Brown | nessuna | Brown è «sheriff-elect» nella contea (1), ma il nostro elenco non contiene una relazione per questa carica. | Alta |

**Altre relazioni considerate ma non assegnate:** per Finland → Helsinki, Iran → Tehran e Scotland → Edinburgh, le frasi non dicono esplicitamente *capitale*; per Queens → Astoria, `/location/neighborhood/neighborhood_of` richiederebbe la direzione inversa; per Betty Smith → Brooklyn, il titolo del libro non dimostra `/people/person/place_lived`; per Sam → Jon, l'annuncio non prova `/people/person/children` tra loro. Il ruolo di sheriff-elect in DeKalb County e la citazione di American Express in un programma NBC non corrispondono a una relazione ammessa per quelle coppie.

**Prima del confronto con i modelli:** le 12 bag danno 6 risposte positive (5 × *contiene*, 1 × *persona–azienda*) e 6 risposte vuote. La concentrazione in due tipi positivi è una proprietà di questo **piccolo campione sotto la regola testuale**, non una restrizione sulle 24 relazioni possibili nel dataset. Alcune scelte restano interpretative, soprattutto Atlanta → Turner Field e Iran → Tehran; il campione serve per capire gli errori, non per stimare l'F1 del dataset. Le predizioni dei modelli non erano state consultate quando le annotazioni sono state fissate.
