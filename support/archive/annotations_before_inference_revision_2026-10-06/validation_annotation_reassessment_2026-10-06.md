# Rivalutazione delle annotazioni: testo, inferenza e conoscenza esterna

6 ottobre 2026. **Stato: proposte da controllare; nessuna etichetta confermata è stata sovrascritta.**

Il criterio scelto dall’autore in chat è: **supporto nel testo, incluse inferenze motivate; fatti esterni distinti**. Ho riletto tutte le 42 bag già annotate (121 frasi fornite): 12 storiche B, 12 C e 18 N. Le C+N costituiscono le 30 del confronto attuale. Il pacchetto A di 20 bag è ancora privo di annotazioni e non rientra in questa revisione. Il riesame usa i testi e le annotazioni salvate, senza aprire nuove predizioni, etichette distanti o etichette ufficiali del test.

**Esito proposto:** mantenere 40 set; discutere due aggiunte inferenziali, B08 `contains` e N18 `nationality`. Entrambe richiedono il check umano, soprattutto B08. Le fonti originali, i loro hash e tutte le decisioni sono nel [JSON della revisione](/Users/alessandro/Progetti/Ensembling/support/validation_annotation_reassessment_2026-10-06.json). Il JSON è una bozza e non è collegato ai programmi di scoring.

## Che cosa stabilisce la letteratura

Il riferimento principale è **Gao et al.**, perché definisce proprio il test manuale di NYT10m. Gli altri lavori aiutano a distinguere inferenza contestuale e recupero di fatti, ma non forniscono una guida esaustiva per le nostre 24 relazioni.

| Paper nella bibliografia | Passaggi verificati | Conseguenza metodologica |
|---|---|---|
| Gao et al., *Manual Evaluation Matters* | §3.1, pp. PDF 3–4; §5.2, p. 6; §5.3, p. 7. [Fonte primaria](https://aclanthology.org/2021.findings-acl.112.pdf) | Gli annotatori giudicano le relazioni espresse da ciascuna frase; il gold della bag è la loro unione. Il paper considera falso positivo un fatto vero nel KG ma non supportato dal testo: Schnitzler nato a Vienna non è estraibile da un racconto ambientato a Vienna. Attribuire questi errori alla conoscenza pre-addestrata è un’ipotesi degli autori, non una prova dell’origine di ogni predizione. |
| Zhang et al., *Rethinking the Role of LLMs for DocRE* | §3.3, p. PDF 4; §5.7, p. 9; Appendix F, pp. 13–15. [Fonte primaria](https://aclanthology.org/2025.naacl-long.319.pdf) | La tendenza degli LLM a scegliere relazioni quando manca evidenza è discussa come errore rispetto al riferimento, anche quando il modello usa conoscenza propria. Non adottano una regola che renda corretto ogni fatto plausibile o vero. |
| Tang et al., *Language model collaboration…* | §3.2, p. PDF 6; §5.9, p. 17; §5.10, p. 20. [Fonte primaria](https://doi.org/10.1016/j.ipm.2025.104286) | Le relazioni implicite sono parte del problema: dal testo che presenta due persone come figli dello stesso padre si può ricavare che sono fratelli. È ragionamento su premesse testuali. Non ho trovato una regola che autorizzi l’aggiunta indiscriminata di fatti storici noti al modello. |
| Chanthran et al., *DocZSRE-SI* | §§3.1–3.2, pp. PDF 2–5; §5.3, p. 8. [Fonte primaria](https://aclanthology.org/2026.eacl-long.216.pdf) | Le descrizioni generate comprendono contesto e attributi impliciti delle entità. Il loro uso è una scelta del metodo; non dimostra che il gold venga esteso a tutti i fatti contenuti nella memoria del generatore. |
| Li et al., ensemble biomedico | Methods/Datasets, p. PDF 2, p. editoriale 1905. [Fonte primaria](https://academic.oup.com/jamia/article/31/9/1904/7634192) | La distinzione novel/background concerne la relazione descritta nell’abstract. Background knowledge qui non significa autorizzare fatti assenti dall’abstract. |
| Han et al., *OpenNRE* | §§2.3–2.4, pp. PDF 2–3. [Fonte primaria](https://aclanthology.org/D19-3029.pdf) | Le bag aggregano evidenze delle frasi. Per DocRE mostrano anche inferenze fra frasi; il toolkit non stabilisce il criterio dettagliato di annotazione del successivo test NYT10m. |
| Christou & Tsoumakas, *Sub-Billion, Super-Frontier* | Appendici A–B e F. [Fonte primaria](https://arxiv.org/html/2606.22606) | I prompt interrogano la relazione nella frase, anche implicita, e le risposte sono confrontate con le etichette dei dataset. Non descrivono una rietichettatura che premi automaticamente conoscenza esterna corretta. |

Gli altri sei lavori del riepilogo — Mojarradi, Cho, survey Li/Wang, Folino, Chan e Lu — riguardano NLU, localizzazione di errori software, sistemi multi-agente, fake news, combinazione generativa o trustworthiness. I loro protocolli non decidono quali relazioni debbano essere annotate nelle frasi di NYT10m. Non ho usato la correttezza fattuale delle risposte in un benchmark di domande come criterio trasferibile automaticamente alla RE.

**Limite della verifica:** nessuno dei passaggi controllati risolve esplicitamente Barzagli → Italy o prescrive una soglia universale fra inferenza pragmatica e conoscenza esterna. Gao richiede supporto testuale, ma questo non equivale a richiedere sempre la parola del nome della relazione. Il protocollo operativo seguente è una nostra proposta dichiarata.

## Criterio operativo proposto

Per ciascuna relazione annotare la premessa testuale, il passaggio inferenziale se esiste, la direzione e l’incertezza. La verità del fatto e il supporto nel testo sono due proprietà diverse; dalla sola risposta del modello non possiamo sapere se abbia ricordato o inferito il fatto.

| Classe | Trattamento | Esempio |
|---|---|---|
| **E: espresso nel testo** | Includere. Sono ammesse normalizzazione linguistica, apposizioni, coreferenza locale e risoluzione dei ruoli. | Chinese director → nationality; father of → children; died at a hospice in X → place_of_death X. |
| **I: inferenza motivata** | Includere solo dichiarando premesse e regola; distinguere implicazioni semantiche da assunzioni pragmatiche. Le nuove assunzioni dubbie vanno decise prima dello scoring. | La convenzione capital → contains è già dichiarata; nazionale che schiera un giocatore → nationality è una nuova convenzione pragmatica da controllare. |
| **K: fatto esterno** | Registrare separatamente, con fonte e data se verificato; non aggiungerlo al gold testuale solo perché è vero. | Nascita/residenza dell’autrice non derivano dal titolo del suo libro; city in country non identifica la capitale. |
| **N: nessuna relazione ammessa supportata** | Proporre `[]`, esplicitando il motivo. Non significa che le entità non abbiano alcuna relazione nel mondo. | Vicinanza geografica senza appartenenza; ruolo non presente nello schema; direzione inversa. |

Una domanda utile è: **se non conoscessi la biografia o la geografia di queste entità, potrei spiegare la relazione usando questa frase e una regola generale dichiarata?** Non è una prova formale, ma rende visibile quando la premessa mancante è un fatto specifico. L’anonimizzazione dei nomi può aiutare a controllare il ragionamento dell’annotatore; non certifica il processo interno del modello.

Accettiamo le normali inferenze spaziali e semantiche già implicite nelle annotazioni precedenti (B01, C01, C04, N01). Per quelle pragmatiche fissiamo una convenzione generale, la applichiamo a tutti i casi compatibili e segnaliamo le alternative. Non basta che un’associazione sia probabile: near → contains, board member → major shareholder, inventor → company founder e birthplace → residence restano passaggi non giustificati.

I vincoli dell’ontologia rimangono quelli del contratto attuale: `children` genitore → figlio; `business/location` azienda → luogo; `contains` luogo → luogo. Luogo → azienda non diventa contains solo perché un edificio commerciale si trova lì. La distinzione fra azienda, prodotto, edificio e persona va risolta prima di assegnare una relazione.

## Due proposte da controllare

### N18 — Andrea Barzagli → Italy

> So Italy will use 25-year-old Andrea Barzagli in Nesta 's spot .

**Prima:** `[]`. **Proposta:** `/people/person/nationality`. **Tipo:** I, inferenza pragmatica; confidenza media.

La motivazione è il ruolo testuale: leggiamo Italy come nazionale che schiera un giocatore al posto di un altro. Ammettendo la convenzione “rappresentanza da giocatore di una nazionale identificabile → nationality del paese rappresentato”, la predizione è sostenibile senza richiamare una biografia individuale. La frase isolata richiede comunque una lettura sportiva e la relazione non è una dichiarazione esplicita di cittadinanza. Questa convenzione non si applica a chi gioca in un club italiano, lavora in Italia, è nato lì o allena la nazionale; non implica una nazionalità esclusiva e richiede attenzione per federazioni/territori che non corrispondono a una nazionalità autonoma.

La [UEFA](https://www.uefa.com/uefachampionsleague/news/0221-0e90d96734ec-096e37359e67-1000--barzagli-doubtful-for-champions-league-final/) lo descrive come Italian international: è una corroborazione esterna distinta, non la premessa testuale. Non presento la convenzione proposta come una regola verificata del manuale NYT10m o come una ricostruzione delle norme di eleggibilità vigenti nel 2006. Se si preferisce escludere questa assunzione di dominio dal gold testuale, l’alternativa resta `[]`, annotando la correttezza fattuale a parte.

### B08 — Iran → Tehran

**Prima:** `[]`. **Proposta da discutere:** `/location/location/contains`. **Tipo:** I, inferenza pragmatica; confidenza medio-bassa. **Alternativa ragionevole:** mantenere `[]`.

Le frasi 5, 11, 13, 18 e 20 collegano luoghi, governo iraniano, rappresentanze e funzionari del paese. Nel loro insieme suggeriscono una localizzazione interna all’Iran. Il salto resta contestabile: un governo può ricevere una proposta all’estero e il nome di una città può designare metonimicamente un governo. Non basta la metonimia per dedurre né contains né capital. La proposta contains usa la convergenza di indizi locativi e istituzionali, da controllare prima di accettarla. **Non propongo capital**, perché lo status di capitale è una premessa ulteriore.

Questa bag è storica e riguarda 20 frasi selezionate su una bag più ampia: il riesame non certifica l’assenza di altre relazioni nelle frasi non mostrate. Non modifica la coorte C+N di 30 bag.

## Revisione completa delle 42 bag

Le relazioni nella tabella sono abbreviazioni; gli ID completi sono nel JSON e nei dettagli. “Mantenere” è una proposta della revisione, non una nuova conferma umana. E/I/N descrivono il supporto dell’intero set; E+I distingue la capitale esplicita dalla sua containment derivata.


| ID | Coppia ordinata | Set precedente | Set proposto | Supporto | Esito |
|---|---|---|---|---|---|
| B01 | Atlanta → Turner Field | `contains` | `contains` | I | Mantenere |
| B02 | Honolulu → Hawaii Kai | `[]` | `[]` | N | Mantenere |
| B03 | NBC → American Express | `[]` | `[]` | N | Mantenere |
| B04 | Jeffrey Katzenberg → DreamWorks Animation | `company` | `company` | E | Mantenere |
| B05 | Queens → Astoria | `contains` | `contains` | E | Mantenere |
| B06 | Betty Smith → Brooklyn | `[]` | `[]` | N | Mantenere |
| B07 | Finland → Helsinki | `contains` | `contains` | E | Mantenere |
| B08 | Iran → Tehran | `[]` | `contains` | I | Aggiunta da controllare |
| B09 | New York → Queens | `contains` | `contains` | E | Mantenere |
| B10 | Sam → Jon | `[]` | `[]` | N | Mantenere |
| B11 | Scotland → Edinburgh | `contains` | `contains` | E | Mantenere |
| B12 | DeKalb County → Derwin Brown | `[]` | `[]` | N | Mantenere |
| C01 | Papua New Guinea → Port Moresby | `capital`, `contains` | `capital`, `contains` | E+I | Mantenere |
| C02 | Voltaire → Paris | `[]` | `[]` | N | Mantenere |
| C03 | West Side → Wal-Mart | `[]` | `[]` | N | Mantenere |
| C04 | Brazil → Brasília | `capital`, `contains` | `capital`, `contains` | E+I | Mantenere |
| C05 | Carlos Slim Helú → Carlos Slim Domit | `children` | `children` | E | Mantenere |
| C06 | Evo Morales → Aymara | `ethnicity` | `ethnicity` | E | Mantenere |
| C07 | Lowry Mays → Mark Mays | `children` | `children` | E | Mantenere |
| C08 | Matt Damon → Leonardo DiCaprio | `[]` | `[]` | N | Mantenere |
| C09 | Ismail Merchant → Mumbai | `place_of_burial`, `place_of_birth` | `place_of_burial`, `place_of_birth` | E | Mantenere |
| C10 | Russia → Saudi Arabia | `[]` | `[]` | N | Mantenere |
| C11 | Mo Mowlam → Canterbury | `place_of_death` | `place_of_death` | E | Mantenere |
| C12 | Facebook → Mark Zuckerberg | `founders` | `founders` | E | Mantenere |
| N01 | Montserrat → Plymouth | `capital`, `contains` | `capital`, `contains` | E+I | Mantenere |
| N02 | Canada → Mississauga | `contains` | `contains` | E | Mantenere |
| N03 | Mike Jeffries → Abercrombie & Fitch | `company` | `company` | E | Mantenere |
| N04 | Jane Elizabeth Hodgson → Crookston | `place_of_birth` | `place_of_birth` | E | Mantenere |
| N05 | Hrishikesh Mukherjee → Mumbai | `place_of_death` | `place_of_death` | E | Mantenere |
| N06 | Lloyd Maines → Natalie Maines | `children` | `children` | E | Mantenere |
| N07 | Lloyd G. Trotter → African American | `ethnicity` | `ethnicity` | E | Mantenere |
| N08 | Krispy Kreme → Winston-Salem | `location` | `location` | E | Mantenere |
| N09 | Calvin Klein → Barry Schwartz | `founders` | `founders` | E | Mantenere |
| N10 | Lou Ye → China | `nationality` | `nationality` | E | Mantenere |
| N11 | Theodor Uppman → San Jose | `place_of_birth` | `place_of_birth` | E | Mantenere |
| N12 | James Manning → Providence | `[]` | `[]` | N | Mantenere |
| N13 | Univision → Gustavo Cisneros | `[]` | `[]` | N | Mantenere |
| N14 | Dyson → James Dyson | `[]` | `[]` | N | Mantenere |
| N15 | Woodcliff Lake → Barr | `[]` | `[]` | N | Mantenere |
| N16 | Silver Lake Partners → Seagate Technology | `[]` | `[]` | N | Mantenere |
| N17 | London → Woolwich | `[]` | `[]` | N | Mantenere |
| N18 | Andrea Barzagli → Italy | `[]` | `nationality` | I | Aggiunta da controllare |

## Motivazioni ed evidenze per ogni bag

### B01 — Atlanta → Turner Field

**Proposta:** `contains`. **Supporto:** I. **Confidenza:** medium. **Frasi lette:** 2.

La locuzione iniziale In Atlanta localizza la statua all’ingresso del campo: un’inferenza spaziale locale sostiene contains. Già la vecchia annotazione ammetteva quindi un’inferenza; la regola precedente non era puramente letterale.

**Frasi di evidenza:** 1.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_blind_sample_2026-09-28.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_blind_annotations_2026-09-28.json)

### B02 — Honolulu → Hawaii Kai

**Proposta:** `[]`. **Supporto:** N. **Confidenza:** high. **Frasi lette:** 1.

near Honolulu esprime prossimità; un luogo vicino può essere interno o esterno alla città. Non basta per contains.

**Conoscenza esterna distinta:** Da verificare: appartenenza geografica effettiva di Hawaii Kai a Honolulu.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_blind_sample_2026-09-28.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_blind_annotations_2026-09-28.json)

### B03 — NBC → American Express

**Proposta:** `[]`. **Supporto:** N. **Confidenza:** high. **Frasi lette:** 1.

Product placement e co-occorrenza non identificano una relazione ammessa tra le due aziende.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_blind_sample_2026-09-28.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_blind_annotations_2026-09-28.json)

### B04 — Jeffrey Katzenberg → DreamWorks Animation

**Proposta:** `company`. **Supporto:** E. **Confidenza:** high. **Frasi lette:** 3.

Katzenberg è indicato come chief executive di DreamWorks Animation. Persona → azienda è la direzione corretta; non si aggiungono ruoli aziendali non descritti.

**Frasi di evidenza:** 1, 2, 3.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_blind_sample_2026-09-28.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_blind_annotations_2026-09-28.json)

### B05 — Queens → Astoria

**Proposta:** `contains`. **Supporto:** E. **Confidenza:** high. **Frasi lette:** 20.

Astoria, Queens e la localizzazione di un’attività in Queens, precisamente in Astoria, supportano contains. neighborhood_of avrebbe direzione Astoria → Queens.

**Frasi di evidenza:** 1, 14.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_blind_sample_2026-09-28.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_blind_annotations_2026-09-28.json)

### B06 — Betty Smith → Brooklyn

**Proposta:** `[]`. **Supporto:** N. **Confidenza:** high. **Frasi lette:** 4.

Brooklyn compare nel titolo dell’opera di Betty Smith. Autrice e ambientazione/titolo non implicano nascita, residenza o morte dell’autrice.

**Conoscenza esterna distinta:** Da verificare: biografia di Betty Smith. Da verificare: residenza di Betty Smith.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_blind_sample_2026-09-28.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_blind_annotations_2026-09-28.json)

### B07 — Finland → Helsinki

**Proposta:** `contains`. **Supporto:** E. **Confidenza:** high. **Frasi lette:** 7.

Helsinki, Finland supporta contains. Non identifica Helsinki come capitale; capital richiede un’informazione ulteriore.

**Frasi di evidenza:** 1, 2, 3, 4, 5.

**Conoscenza esterna distinta:** Da verificare: status di Helsinki; non deducibile dalla sola localizzazione.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_blind_sample_2026-09-28.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_blind_annotations_2026-09-28.json)

### B08 — Iran → Tehran

**Proposta:** `contains`. **Supporto:** I. **Confidenza:** low_to_medium. **Frasi lette:** 20.

Proposta dubbia: i riferimenti al governo iraniano a Tehran, all’assenza di una rappresentanza statunitense e ai colloqui con funzionari del paese suggeriscono Tehran come luogo interno all’Iran. È un’inferenza pragmatica del contesto istituzionale, non una deduzione necessaria. La sola metonimia Tehran = governo iraniano non basterebbe. Non aggiungere capital.

**Frasi di evidenza:** 5, 11, 13, 18, 20.

**Conoscenza esterna distinta:** Da verificare: status di Tehran; il contesto istituzionale non lo enuncia.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_blind_sample_2026-09-28.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_blind_annotations_2026-09-28.json)

### B09 — New York → Queens

**Proposta:** `contains`. **Supporto:** E. **Confidenza:** high. **Frasi lette:** 20.

Queens, New York e i quartieri descritti come parte di New York supportano contains. Nessun bisogno di sapere dalla memoria dove si trova Queens.

**Frasi di evidenza:** 12, 13, 14.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_blind_sample_2026-09-28.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_blind_annotations_2026-09-28.json)

### B10 — Sam → Jon

**Proposta:** `[]`. **Supporto:** N. **Confidenza:** high. **Frasi lette:** 1.

Robin/Sam sono figli della persona dell’epigrafe; Melissa/Jon sono nipoti. Non è specificato di quale figlio siano figli: Sam → Jon non è stabilito.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_blind_sample_2026-09-28.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_blind_annotations_2026-09-28.json)

### B11 — Scotland → Edinburgh

**Proposta:** `contains`. **Supporto:** E. **Confidenza:** high. **Frasi lette:** 6.

Edinburgh, Scotland supporta contains. Lo status di capitale non è enunciato e il solo nome del luogo non lo dimostra.

**Frasi di evidenza:** 1, 2, 3, 4, 6.

**Conoscenza esterna distinta:** Da verificare: status di Edinburgh e corretta distinzione country/region.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_blind_sample_2026-09-28.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_blind_annotations_2026-09-28.json)

### B12 — DeKalb County → Derwin Brown

**Proposta:** `[]`. **Supporto:** N. **Confidenza:** high. **Frasi lette:** 1.

La carica di sheriff-elect nella contea non ha una relazione corrispondente nell’inventario per contea → persona. Non si converte la persona in un luogo o in un evento.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_blind_sample_2026-09-28.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_blind_annotations_2026-09-28.json)

### C01 — Papua New Guinea → Port Moresby

**Proposta:** `capital`, `contains`. **Supporto:** E+I. **Confidenza:** high. **Frasi lette:** 1.

Port Moresby è chiamata capitale di Papua New Guinea. contains è l’implicazione/convenzione già dichiarata per capital; conservarla senza nuovi controlli del test.

**Frasi di evidenza:** 1.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_clean_sample_2026-09-29.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_clean_annotations_2026-09-29.json)

### C02 — Voltaire → Paris

**Proposta:** `[]`. **Supporto:** N. **Confidenza:** high. **Frasi lette:** 3.

Le menzioni riguardano un titolo, un ristorante e un confronto con Voltaire. Nessuna relazione biografica Voltaire → Paris emerge. Resta anche un problema di allineamento delle menzioni al personaggio.

**Conoscenza esterna distinta:** Da verificare solo dopo controllo di identità delle menzioni: il nome nel testo non identifica sempre il filosofo.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_clean_sample_2026-09-29.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_clean_annotations_2026-09-29.json)

### C03 — West Side → Wal-Mart

**Proposta:** `[]`. **Supporto:** N. **Confidenza:** high. **Frasi lette:** 1.

Una futura sede commerciale è sul West Side, ma la coppia è luogo → azienda. business/location ha direzione opposta; contains nel contratto attuale è luogo → luogo, non luogo → persona giuridica.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_clean_sample_2026-09-29.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_clean_annotations_2026-09-29.json)

### C04 — Brazil → Brasília

**Proposta:** `capital`, `contains`. **Supporto:** E+I. **Confidenza:** high. **Frasi lette:** 4.

Brasília è definita capitale e capitale amministrativa del Brasile. contains conserva l’implicazione/convenzione dichiarata per capital.

**Frasi di evidenza:** 3, 4.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_clean_sample_2026-09-29.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_clean_annotations_2026-09-29.json)

### C05 — Carlos Slim Helú → Carlos Slim Domit

**Proposta:** `children`. **Supporto:** E. **Confidenza:** high. **Frasi lette:** 1.

eldest son esplicita Helú → Domit come genitore → figlio.

**Frasi di evidenza:** 1.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_clean_sample_2026-09-29.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_clean_annotations_2026-09-29.json)

### C06 — Evo Morales → Aymara

**Proposta:** `ethnicity`. **Supporto:** E. **Confidenza:** high. **Frasi lette:** 1.

Evo Morales è descritto come an Aymara: ethnicity è direttamente supportata.

**Frasi di evidenza:** 1.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_clean_sample_2026-09-29.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_clean_annotations_2026-09-29.json)

### C07 — Lowry Mays → Mark Mays

**Proposta:** `children`. **Supporto:** E. **Confidenza:** high. **Frasi lette:** 1.

Mark Mays è detto son of Lowry Mays: children nella direzione Lowry → Mark, indipendentemente dalle etichette distanti.

**Frasi di evidenza:** 1.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_clean_sample_2026-09-29.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_clean_annotations_2026-09-29.json)

### C08 — Matt Damon → Leonardo DiCaprio

**Proposta:** `[]`. **Supporto:** N. **Confidenza:** high. **Frasi lette:** 1.

Recitare nello stesso film non è una delle relazioni ammesse per Damon → DiCaprio.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_clean_sample_2026-09-29.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_clean_annotations_2026-09-29.json)

### C09 — Ismail Merchant → Mumbai

**Proposta:** `place_of_burial`, `place_of_birth`. **Supporto:** E. **Confidenza:** high. **Frasi lette:** 1.

La frase specifica sepoltura a Mumbai e nascita nello stesso luogo tramite where. Non specifica né morte né residenza.

**Frasi di evidenza:** 1.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_clean_sample_2026-09-29.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_clean_annotations_2026-09-29.json)

### C10 — Russia → Saudi Arabia

**Proposta:** `[]`. **Supporto:** N. **Confidenza:** high. **Frasi lette:** 1.

Russia e Saudi Arabia sono elementi di una lista di paesi; non è espressa una relazione ammessa fra loro.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_clean_sample_2026-09-29.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_clean_annotations_2026-09-29.json)

### C11 — Mo Mowlam → Canterbury

**Proposta:** `place_of_death`. **Supporto:** E. **Confidenza:** high. **Frasi lette:** 1.

died ... at a hospice in Canterbury supporta place_of_death, inclusa la normale risoluzione spaziale ospizio → città.

**Frasi di evidenza:** 1.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_clean_sample_2026-09-29.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_clean_annotations_2026-09-29.json)

### C12 — Facebook → Mark Zuckerberg

**Proposta:** `founders`. **Supporto:** E. **Confidenza:** high. **Frasi lette:** 1.

Zuckerberg è chiamato founder di Facebook: company → founder.

**Frasi di evidenza:** 1.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_clean_sample_2026-09-29.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_clean_annotations_2026-09-29.json)

### N01 — Montserrat → Plymouth

**Proposta:** `capital`, `contains`. **Supporto:** E+I. **Confidenza:** high. **Frasi lette:** 1.

Plymouth è chiamata capitale di Montserrat. Si conservano la classificazione e la convenzione capital/contains già adottate; il passo non giustifica da solo un cambio della tassonomia country/region.

**Frasi di evidenza:** 1.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_2026-10-01.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_annotations_2026-10-01.json)

### N02 — Canada → Mississauga

**Proposta:** `contains`. **Supporto:** E. **Confidenza:** high. **Frasi lette:** 1.

Mississauga, Canada supporta Canada → Mississauga contains.

**Frasi di evidenza:** 1.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_2026-10-01.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_annotations_2026-10-01.json)

### N03 — Mike Jeffries → Abercrombie & Fitch

**Proposta:** `company`. **Supporto:** E. **Confidenza:** high. **Frasi lette:** 1.

chief executive of Abercrombie & Fitch supporta Jeffries → azienda.

**Frasi di evidenza:** 1.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_2026-10-01.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_annotations_2026-10-01.json)

### N04 — Jane Elizabeth Hodgson → Crookston

**Proposta:** `place_of_birth`. **Supporto:** E. **Confidenza:** high. **Frasi lette:** 1.

was born ... in Crookston supporta place_of_birth.

**Frasi di evidenza:** 1.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_2026-10-01.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_annotations_2026-10-01.json)

### N05 — Hrishikesh Mukherjee → Mumbai

**Proposta:** `place_of_death`. **Supporto:** E. **Confidenza:** high. **Frasi lette:** 1.

died ... in Mumbai supporta place_of_death; non basta per nascita o residenza.

**Frasi di evidenza:** 1.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_2026-10-01.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_annotations_2026-10-01.json)

### N06 — Lloyd Maines → Natalie Maines

**Proposta:** `children`. **Supporto:** E. **Confidenza:** high. **Frasi lette:** 1.

father of Natalie Maines supporta Lloyd → Natalie children. Non occorre la parola children nel testo; si normalizza la relazione senza invertirla.

**Frasi di evidenza:** 1.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_2026-10-01.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_annotations_2026-10-01.json)

### N07 — Lloyd G. Trotter → African American

**Proposta:** `ethnicity`. **Supporto:** E. **Confidenza:** high. **Frasi lette:** 1.

an African American supporta ethnicity.

**Frasi di evidenza:** 1.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_2026-10-01.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_annotations_2026-10-01.json)

### N08 — Krispy Kreme → Winston-Salem

**Proposta:** `location`. **Supporto:** E. **Confidenza:** high. **Frasi lette:** 1.

Krispy Kreme’s headquarters in Winston-Salem supporta business/location. Avere qui la sede non implica che qui sia stata fondata.

**Frasi di evidenza:** 1.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_2026-10-01.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_annotations_2026-10-01.json)

### N09 — Calvin Klein → Barry Schwartz

**Proposta:** `founders`. **Supporto:** E. **Confidenza:** medium. **Frasi lette:** 1.

Barry Schwartz è chiamato co-founder of Calvin Klein. La lettura aziendale del nome Calvin Klein è determinata dal ruolo co-founder; mantenere founders e segnalare l’ambiguità di entità.

**Frasi di evidenza:** 1.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_2026-10-01.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_annotations_2026-10-01.json)

### N10 — Lou Ye → China

**Proposta:** `nationality`. **Supporto:** E. **Confidenza:** high. **Frasi lette:** 1.

Chinese director Lou Ye supporta nationality tramite normalizzazione del demònimo Chinese → China. Non occorre la parola nationality.

**Frasi di evidenza:** 1.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_2026-10-01.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_annotations_2026-10-01.json)

### N11 — Theodor Uppman → San Jose

**Proposta:** `place_of_birth`. **Supporto:** E. **Confidenza:** high. **Frasi lette:** 1.

was born in San Jose supporta place_of_birth. Nascita e origine familiare svedese non dimostrano residenza successiva né un’altra relazione per questa coppia.

**Frasi di evidenza:** 1.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_2026-10-01.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_annotations_2026-10-01.json)

### N12 — James Manning → Providence

**Proposta:** `[]`. **Supporto:** N. **Confidenza:** high. **Frasi lette:** 1.

Providence descrive la Brown family; non è localizzato lì James Manning, né la sua morte. Un eventuale fatto biografico esterno resta distinto.

**Conoscenza esterna distinta:** Da verificare: biografia di James Manning e luogo del decesso.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_2026-10-01.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_annotations_2026-10-01.json)

### N13 — Univision → Gustavo Cisneros

**Proposta:** `[]`. **Supporto:** N. **Confidenza:** high. **Frasi lette:** 1.

La frase dice member of Univision’s board. Essere nel consiglio non implica essere un major shareholder; la relazione persona → azienda avrebbe direzione opposta.

**Conoscenza esterna distinta:** Da verificare: partecipazione azionaria, dimensione della quota e data.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_2026-10-01.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_annotations_2026-10-01.json)

### N14 — Dyson → James Dyson

**Proposta:** `[]`. **Supporto:** N. **Confidenza:** high. **Frasi lette:** 1.

Inventore di un prodotto e fondatore di una società sono ruoli diversi. his product line non definisce la fondazione o la struttura proprietaria dell’azienda. Un fatto esterno verificato potrebbe essere vero senza essere estratto qui.

**Conoscenza esterna distinta:** Da verificare: storia della società Dyson; inventore non basta come premessa testuale.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_2026-10-01.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_annotations_2026-10-01.json)

### N15 — Woodcliff Lake → Barr

**Proposta:** `[]`. **Supporto:** N. **Confidenza:** high. **Frasi lette:** 1.

Based in Woodcliff Lake localizza Barr, ma la query è luogo → azienda. Stessa decisione di C03: rispettare business/location azienda → luogo e contains luogo → luogo.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_2026-10-01.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_annotations_2026-10-01.json)

### N16 — Silver Lake Partners → Seagate Technology

**Proposta:** `[]`. **Supporto:** N. **Confidenza:** high. **Frasi lette:** 1.

Il soggetto investitore è Silver Lake e l’impresa acquisita è Seagate. Un’eventuale company → major shareholder sarebbe Seagate → Silver Lake, direzione opposta. Inoltre la frase non quantifica la quota.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_2026-10-01.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_annotations_2026-10-01.json)

### N17 — London → Woolwich

**Proposta:** `[]`. **Supporto:** N. **Confidenza:** high. **Frasi lette:** 1.

about 10 miles east of central London dà una distanza dal centro; non stabilisce un confine amministrativo. L’appartenenza effettiva di Woolwich a London è un fatto geografico ulteriore.

**Conoscenza esterna distinta:** Da verificare: appartenenza geografica e livello territoriale London/Woolwich.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_2026-10-01.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_annotations_2026-10-01.json)

### N18 — Andrea Barzagli → Italy

**Proposta:** `nationality`. **Supporto:** I. **Confidenza:** medium. **Frasi lette:** 1.

Proporre nationality come inferenza pragmatica: Italy will use ... in Nesta’s spot è letto come la nazionale che schiera il giocatore. La convenzione ammessa collega la rappresentanza da giocatore di una nazionale identificabile al paese rappresentato. È un’assunzione di dominio esplicita, non una citazione di cittadinanza: non estenderla a club, allenatori, residenza o nascita. Una lettura più prudente della frase isolata può mantenere [].

**Frasi di evidenza:** 1.

**Conoscenza esterna distinta:** UEFA descrive Barzagli come Italian international. La fonte è esterna e non costituisce la premessa dell’annotazione testuale proposta.

[Testo completo della bag](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_2026-10-01.json) · [Annotazione precedente](/Users/alessandro/Progetti/Ensembling/support/validation_expanded_18_annotations_2026-10-01.json)

## Dopo il controllo umano

Le nuove proposte rimangono separate dai riferimenti accettati. Dopo il controllo umano si potrà creare una versione successiva delle annotazioni, aggiornare la regola nel protocollo e le descrizioni nei report, mantenendo gli originali e distinguendo i risultati storici da quelli rivalutati. Per lo scoring operativo occorrono le decisioni definitive su B08/N18; una bozza incerta non va usata come nuovo gold.

Il test ufficiale conserva le etichette rilasciate e il loro criterio di aggregazione. Questa revisione di sviluppo non autorizza a rietichettare il test per premiare i nostri modelli. Eventuali discrepanze semantiche del benchmark possono essere discusse separatamente. Le 42 bag sono selezionate e già usate nello sviluppo, quindi non forniscono una stima indipendente o rappresentativa delle prestazioni.

Una valutazione aggiuntiva della verità nel mondo richiederebbe fonti datate, identificazione delle entità, verifica delle predizioni e attenzione alla completezza del riferimento. Controllare soltanto i fatti predetti potrebbe sostenere un’analisi di precisione fattuale, ma non un richiamo o F1 fattuale completo. Qui i candidati K non corroborati sono esplicitamente da verificare; non sono un secondo gold.

## PDF locali usati per i passaggi metodologici

:codex-file-citation{path="/Users/alessandro/Library/CloudStorage/OneDrive-UniversitàdegliStudidiMilano/unimi/tesi/bibliografia/Tianyu Gao et al.pdf" purpose="source"}

:codex-file-citation{path="/Users/alessandro/Library/CloudStorage/OneDrive-UniversitàdegliStudidiMilano/unimi/tesi/bibliografia/Fu Zhang et al.pdf" purpose="source"}

:codex-file-citation{path="/Users/alessandro/Library/CloudStorage/OneDrive-UniversitàdegliStudidiMilano/unimi/tesi/bibliografia/Tang et al.pdf" purpose="source"}

:codex-file-citation{path="/Users/alessandro/Library/CloudStorage/OneDrive-UniversitàdegliStudidiMilano/unimi/tesi/bibliografia/Chanthran et al.pdf" purpose="source"}

:codex-file-citation{path="/Users/alessandro/Library/CloudStorage/OneDrive-UniversitàdegliStudidiMilano/unimi/tesi/bibliografia/Zhao Li-Qiang Wei.pdf" purpose="source"}

:codex-file-citation{path="/Users/alessandro/Library/CloudStorage/OneDrive-UniversitàdegliStudidiMilano/unimi/tesi/bibliografia/Xu Han et al.pdf" purpose="source"}

## Verifiche della revisione

- 42 coppie distinte; 121 frasi nei tre campioni originali; copertura di tutte le annotazioni salvate.
- Hash di ogni campione coerente con il riferimento originale; hash delle tre annotazioni originali salvati nel JSON.
- Tutti gli ID proposti appartengono all’inventario positivo NYT10m; numeri di frase validi.
- 40 proposte di mantenimento, 2 proposte di aggiunta; nessuna conferma umana delle nuove etichette e nessuno scoring ricalcolato.
- Nessuna nuova ispezione del test ufficiale o delle predizioni durante il riesame.
