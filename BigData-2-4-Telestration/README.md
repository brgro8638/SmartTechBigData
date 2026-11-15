# Telestrasjon på endoskopivideo (Big Data, oppgave 2-4)

Denne teksten er for deg som kan programmering og litt bildebehandling, men som ikke kjenner til
telestrasjon eller sporingsmetodene vi har brukt. Den forklarer hva appen gjør, hvilke parametere
som finnes, hva de styrer og hva du kan vente deg når du endrer dem.

```bash
pip install -r requirements.txt
streamlit run telestration_app.py
```

Legg videoene i `videos/` (gitignored, siden repoet er offentlig) eller last opp en fil i sidepanelet.

---

## 1. Hva er telestrasjon?

*Telestrasjon* betyr å tegne oppå levende video. Det kjenner du fra fotball-TV, der en ekspert
tegner piler på banen. I kirurgi brukes det til undervisning og veiledning: en erfaren kirurg tegner
for eksempel en linje langs der et snitt skal gå, på skjermen som viser bildet fra laparoskopet
(kameraet inne i buken).

Problemet er at **kameraet beveger seg hele tiden**. Kirurgen eller assistenten flytter, roterer og
zoomer med skopet. En tegning som bare ligger fast i skjermkoordinater, havner raskt på feil vev.
Det kaller oppgaven *camera moving problem*. Tegningen må altså **følge vevet**, ikke skjermen.

## 2. Hva appen gjør

1. Du velger en video, et startbilde og en klipplengde (standard 5 s).
2. Du klikker 2 punkter (linje A→B) eller 3 punkter (knekt linje A→B→C) på startbildet.
3. Appen leser klippet, skalerer det ned til 640 px bredde og kjører **tre ulike sporingsmetoder**
   (trackere). Hver av dem flytter tegningen bilde for bilde.
4. Resultatet er en video der alle tre tegningene vises i hver sin farge, i tillegg til en tynn hvit
   linje som viser hvor tegningen *ville stått uten sporing*. Til slutt kommer grafer og en
   automatisk diskusjon.

Alt kjøres **to ganger**: én gang med preprosessering og én gang uten. Da kan grafene vise hvilken
effekt preprosesseringen faktisk har.

```
videobilde ─► nedskalering til 640 px ─► gråtone ─► [CLAHE] ─► [glare-maske + inpainting] ─► [kantmaske]
                                                                          │
                     ┌────────────────────────────────────────────────────┼─────────────────────┐
                     ▼                                                    ▼                     ▼
          A) Lucas–Kanade på de                          B) ORB-punkter + matching       C) Fasekorrelasjon
             klikkede punktene                              + RANSAC-homografi             (kun forskyvning)
                     └──────────────► nye posisjoner for A, B, C ◄────────────────────────────────┘
                                                      │
                                   tegnes på originalvideoen (i farger)
```

Preprosesseringen brukes **bare på gråtonekopien som trackerne ser**. Videoen du ser, er alltid den
originale fargevideoen.

## 3. De tre trackerne

### A) Lucas–Kanade (LK), cyan
Følger **akkurat de pikslene du klikket på**. For hvert punkt ser den på et lite vindu rundt punktet
(31×31 px) og finner hvor det samme mønsteret har flyttet seg i neste bilde. Den bygger på tre
antakelser:
- **Lysstyrken er konstant** (samme vevsbit ser lik ut fra bilde til bilde).
- **Bevegelsen er liten** mellom to bilder.
- **Nabopiksler beveger seg likt.**

For å takle større bevegelser bruker den en **bildepyramide**: først et grovt estimat på et kraftig
nedskalert bilde, så finjustering nivå for nivå (coarse-to-fine).

**Forward–backward-sjekk:** vi sporer punktet fremover til neste bilde og så bakover igjen. Hvis vi
ikke lander der vi startet, har antakelsene sviktet, og punktet regnes som *tapt*. Et tapt punkt
fryses der det sist var bra og markeres med en rød X i videoen.

- ✅ Veldig rask (< 1 ms per bilde) og veldig nøyaktig så lenge punktet ligger på tekstur.
- ❌ Klikker du på glatt vev uten tekstur, finner den ikke noe å feste seg til. Rask bevegelse, røyk,
  blod eller et instrument som passerer kan få den til å miste punktet.

### B) ORB + RANSAC-homografi, grønn
Bryr seg ikke om de klikkede punktene. Den finner i stedet **hele kamerabevegelsen** fra hele
bildet:
1. **ORB** finner opptil 1000 karakteristiske punkter (hjørner) i hvert bilde og lager en binær
   «fingeravtrykk»-beskrivelse av hvert av dem.
2. **Matching:** hvert punkt i forrige bilde pares med det mest like punktet i nåværende bilde.
   **Lowe ratio test** kaster parene der nest beste kandidat er nesten like god, fordi slike par er
   tvetydige (typisk ved repeterende slimhinnemønster).
3. **RANSAC** prøver gjentatte ganger å forklare bevegelsen ut fra 4 tilfeldige par, og beholder den
   modellen flest par er enige om. Feilmatcher og ting som beveger seg på egen hånd (instrumenter)
   blir *outliers*.
4. Modellen er en **homografi**, en 3×3-matrise som kan beskrive forskyvning, rotasjon, zoom og
   perspektiv. Homografiene fra bilde til bilde ganges sammen, og tegningen transformeres med
   resultatet.

- ✅ Mest robust, fordi den ikke er avhengig av at akkurat dine punkter har tekstur.
- ❌ En homografi antar at motivet er **plant**, og det er buken ikke. Når mange små feil ganges sammen
  over mange bilder, oppstår det **drift**.

### C) Fasekorrelasjon, oransje
Sammenligner to hele bilder i **frekvensdomenet** (Fourier-transform). En ren forskyvning av
bildet gir en topp i den inverse transformen, og plasseringen av toppen er forskyvningen (dx, dy).
Alle punktene i tegningen flyttes like mye.

- ✅ Rask, robust mot støy og lysendringer.
- ❌ Gir **bare forskyvning**. Zoom eller rotasjon av skopet blir ikke fanget opp, så tegningen
  hverken skaleres eller roteres.

### Hvorfor ikke YOLO?
YOLO finner *objekter* (for eksempel instrumenter) og tegner bokser rundt dem. Det følger ikke *vev*.
YOLO er derfor nevnt som fremtidig arbeid: til å maskere bort instrumenter, eller til å feste en
tegning til instrumentspissen.

## 4. Preprosessering

| Steg | Hva det gjør | Hvorfor |
|---|---|---|
| **CLAHE** (Contrast Limited Adaptive Histogram Equalization) | Deler bildet i ruter og strekker kontrasten i hver rute for seg, med et tak («clip limit») så støy ikke forsterkes ubegrenset. | Endoskopisk vev er lavkontrast. Mer lokal kontrast gir flere hjørner å spore. På vår video økte antall hjørner fra ca. 1000 til 3200 per bilde. |
| **Glare-maske** | Terskel på svært lyse piksler, deretter *dilasjon* (morfologi) som gjør hver flekk litt større. | Glare er speilrefleks av lyset på skopet. Den følger **lyset/kameraet, ikke vevet**, så hjørner i glare gir falske spor. |
| **Inpainting** | Fyller glare-områdene med verdier fra vevet rundt. | LK og fasekorrelasjon kan ikke bruke en maske, fordi de ser på alle pikslene i et vindu. En glare-flekk i LK-vinduet trekker punktet med seg. |
| **Kantmaske** | Masker bort svart ramme og tekst rundt endoskopbildet. | Rammen står stille i forhold til kameraet og trekker estimatet mot «ingen bevegelse». |

Vi bruker bevisst **ikke** *opening* (erosjon + dilasjon). På vått fettvev består glare av små
prikker på 1–3 px, og opening fjernet nettopp dem som «støy». Da ble masken tom.

**Målt effekt på en syntetisk testvideo med fasit** (gjennomsnittlig feil over 5 s):

| | Uten preprosessering | Med preprosessering |
|---|---|---|
| A) Lucas–Kanade | 8.6 px | 0.4 px |
| B) ORB + RANSAC | 18.6 px | 10.1 px |
| C) Fasekorrelasjon | 16.2 px | 9.4 px |

## 5. Parametere: hva de styrer og hva du kan vente deg

### Video og tegning (sidepanel, del 1)

| Parameter | Standard | Styrer | Når du endrer |
|---|---|---|---|
| **Video folder / upload** | `videos/` | Hvor videoene hentes fra. | – |
| **Clip length (s)** | 5 (2–10) | Hvor lenge tegningen skal spores. | **Lengre:** mer drift i ORB og fasekorrelasjon (feilene akkumuleres), større sjanse for at LK mister punkter, og lineært lengre kjøretid. **Kortere:** alt ser bedre ut, men demonstrerer mindre. |
| **Jump to (s)** / **Start frame** | 5 s | Bildet du «pauser» på og tegner på. Feltet hopper grovt, slideren finjusterer. | Det avgjør hvilken scene du tester. Rolig kamera og lite instrumenter gir pene resultater (f.eks. 402 s). Rask bevegelse gir feil (f.eks. 382 s, der LK mister alle punktene). |
| **Drawing** | Linje (2 punkter) | 2 eller 3 punkter. | Flere punkter gir flere sjanser for at LK mister ett. Med fasekorrelasjon beholder tegningen alltid nøyaktig samme form (bare forskyvning). Med ORB kan den rotere, skaleres og forvrenges. |

**Hvor du klikker** er i praksis den viktigste «parameteren» for LK: klikk på tydelige strukturer
som kar, kanter mellom fett og bindevev eller små blødningsflekker. Unngå glatt, jevnt vev og glare.

### Preprosessering (sidepanel, del 2)

| Parameter | Standard | Styrer | Når du endrer |
|---|---|---|---|
| **CLAHE** av/på | på | Om kontrastforsterkning brukes. | **Av:** færre hjørner (preview viser antallet). ORB får færre matcher, og LK får svakere gradienter i glatt vev. |
| **CLAHE clipLimit** | 2.0 (1–6) | Taket for hvor mye kontrasten får strekkes. | **Høyere:** mer kontrast og flere hjørner, men også mer forsterket støy og kompresjonsartefakter. Det kan gi flere tvetydige matcher og litt lavere inlier-ratio. **1.0:** nesten ingen effekt. |
| **CLAHE tileGridSize** | 8 (2–16) | Antall ruter per akse. Bildet deles i 8×8 ruter. | **Høyere** (mindre ruter): mer lokal utjevning, som får frem små detaljer men kan gi synlige rutegrenser og mer støy. **Lavere:** nærmer seg vanlig global histogramutjevning. |
| **Specular-highlight mask** av/på | på | Glare-maske og inpainting. | **Av:** hjørner på glare kommer tilbake, og glare-flekker i LK-vinduet kan dra punkter med kameraet. Effekten er størst i scener med mye vått vev. |
| **Glare threshold** | 220 (180–254) | Gråverdien som regnes som glare. | **Lavere:** mer av bildet maskeres. Lyst fett kan bli feiltolket som glare, og da forsvinner ekte vevsdetaljer og blir inpaintet. Preview viser hvor stor andel som maskeres i rødt. **Høyere:** bare de aller hvitest pikslene. Den lysende «glorien» rundt refleksen blir med, og den gir mange falske hjørner. (Med 230 og opening fikk vi 0 % maskert på denne videoen.) |
| **Mask dilation (px)** | 5 (0–20) | Hvor mye hver glare-flekk utvides. | **Større:** dekker glorien og slår sammen nærliggende prikker, men fjerner mer vev. Inpainting blir tregere og mindre troverdig over store områder. **0:** bare kjernen maskeres, og kanten av refleksen gir fortsatt hjørner. |
| **Mask black border / overlays** | på | Masker svart ramme og tekst rundt bildet. | Liten effekt på vår video, som fyller hele bildet. Avgjørende på videoer med rund svart ramme eller tidsstempel: uten maske undervurderer ORB og fasekorrelasjon bevegelsen. |

### Sporing (sidepanel, del 3)

| Parameter | Standard | Styrer | Når du endrer |
|---|---|---|---|
| **LK forward-backward limit (px)** | 4.0 (0.5–10) | Hvor langt et LK-punkt kan bomme på tur–retur før det regnes som tapt. | **Lavere:** strengere, så punkter erklæres tapt tidligere. Med 2 px mistet vi punkter unødvendig på grunn av enkeltbilder med små hopp. **Høyere:** punkter beholdes lenger, men et punkt som har sklidd av kan bli stående som «gyldig». |

### Visning (hovedsiden)

| Parameter | Styrer |
|---|---|
| **Trackers shown in the video** | Hvilke trackere som tegnes i videoen. Påvirker bare visningen, ikke beregningen. |
| **Frame** (slider) | Bla bilde for bilde i resultatet. Nyttig for å finne nøyaktig når en tracker feiler. |

### Konstanter i koden (for den som vil eksperimentere)

| Konstant | Verdi | Når du endrer |
|---|---|---|
| `PROC_WIDTH` | 640 | Høyere gir mer detaljer og presisjon, men tregere kjøring (omtrent kvadratisk med bredden). Lavere er raskere, men små strukturer forsvinner. |
| `LK_PARAMS["winSize"]` | 31×31 | Større vindu er mer robust i vev med lite tekstur, men blander inn naboområder som beveger seg annerledes, og fanger lettere opp glare. Mindre vindu er mer presist, men taper oftere punkter. |
| `LK_PARAMS["maxLevel"]` | 3 | Antall pyramidenivåer. Flere nivåer takler større bevegelser per bilde. Færre gjør LK sårbar for rask bevegelse. |
| `LOWE_RATIO` | 0.75 | Høyere (f.eks. 0.9) gir flere matcher, men flere gale. Lavere (0.6) gir færre, men sikrere matcher. Blir det under 8, hoppes bildet over. |
| `RANSAC_REPROJ_PX` | 3.0 | Hvor nær et par må passe modellen for å være inlier. Høyere gir høyere inlier-ratio, men mindre presis homografi. Lavere er strengere. |
| `ORB_create(nfeatures=1000, fastThreshold=12)` | | Flere punkter eller lavere terskel gir flere matcher og mer robusthet, men tregere ORB (allerede den tregeste med ca. 9 ms per bilde). |
| `plausible_homography` | det 0.7–1.4 | Forkaster homografier som er urealistiske mellom to bilder (for mye zoom eller perspektiv). Da blir tegningen stående i det bildet i stedet for å «hoppe». |

## 6. Hvordan lese resultatene

**Tallene øverst:**
- *LK points kept*: antall punkter LK fortsatt har på slutten, med endringen mot kjøringen uten
  preprosessering.
- *ORB valid homographies*: andelen bilder der ORB fant en brukbar bevegelsesmodell.
- *ORB mean inlier ratio*: andelen matcher som stemte med modellen. Over 0.8 er bra.
- *Camera motion*: hvor langt tegningen flyttet seg ifølge ORB.

**Grafene:**
1. **Displacement:** hvor langt hver trackers tegning har flyttet seg fra start. Kurver som følger
   hverandre betyr at trackerne er enige. Den røde stiplede linjen er avstanden mellom LK og ORB.
   Øker den jevnt, driver én av dem. Hopper den, har LK mistet et punkt.
2. **Good features:** antall FAST-hjørner per bilde: rått bilde (grå), etter preprosessering (grønn)
   og hjørner som lå på glare eller ramme og ble kastet (rød).
3. **LK:** forward–backward-feil per bilde og antall tapte punkter. Topper i feilen viser hvor
   antakelsene ble brutt.
4. **ORB:** inlier-ratio per bilde. Hull betyr at ingen gyldig homografi ble funnet, og et fall betyr
   at bildet var vanskelig (bevegelsesuskarphet, røyk, instrumenter).
5. **Processing time:** gjennomsnittlig tid per bilde per steg, med 40 ms-grensen for sanntid ved 25 fps.

**Begrensning:** på ekte video har vi ingen fasit. Vi måler derfor *indirekte* kvalitet: om
trackerne er enige, forward–backward-feil og inlier-ratio. Den eksakte nøyaktigheten er målt på den
syntetiske testvideoen.
