# DTMM 3-ias laboratorinis darbas

A25 duomenų dimensijos mažinimas, klasterizavimas ir klasterių kokybės bei stabilumo vertinimas.

## Reikalavimai

- Python 3 su `numpy` ir `pillow` bibliotekomis.
- R su terminale pasiekiama `Rscript` komanda. Papildomų R paketų nereikia.
- `A25_clean.csv` ir `A25_normuotas.csv` failai tame pačiame aplanke kaip skriptai. Juos reikia pridėti atskirai.

Atidarykite terminalą repozitorijos aplanke. Jei Python aplinka dar nesukurta, paleiskite šias komandas:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pillow
```

## 5 skyrius. Dimensijos mažinimas

Apskaičiuoja PCA bei t-SNE koordinates ir sukuria projekcijų grafiką. Rezultatai išsaugomi aplanke `rezultatai_5`.

```bash
python A25_5_dimensijos_mazinimas.py \
  A25_normuotas.csv rezultatai_5
```

## 6 skyrius. Klasterizavimas

Paleiskite po 5 skyriaus. Skriptas taiko „k-means“, Ward ir DBSCAN metodus pradinei bei sumažintoms erdvėms. Klasterių priskyrimai, suvestinės ir grafikas išsaugomi aplanke `rezultatai_6`.

```bash
Rscript A25_6_klasterizavimas.R \
  A25_normuotas.csv \
  rezultatai_5/A25_5_koordinates.csv \
  rezultatai_6
```


## 7 skyrius. Vertinimas

Paleiskite po 5 ir 6 skyrių. Skriptas apskaičiuoja klasterių kokybės matus ir įvertina stabilumą dešimtyje kartotinių imčių. Rezultatų lentelės išsaugomos aplanke `rezultatai_7`.

```bash
python A25_7_vertinimas.py \
  A25_clean.csv \
  A25_normuotas.csv \
  rezultatai_5/A25_5_koordinates.csv \
  rezultatai_6/A25_6_priskyrimai.csv \
  A25_5_dimensijos_mazinimas.py \
  A25_6_klasterizavimas.R \
  rezultatai_7
```

### 7.2 poskyris. ARI grafikas

Paleiskite po 7 skyriaus. Skriptas braižo vidutinio ARI priklausomybę nuo imčių skaičiaus. Pradinės ir PCA erdvių rezultatus perskaičiuoja iki 20 imčių, o t-SNE rezultatus skaito iš `rezultatai_7/A25_7_imciu_poros.csv`.

Reikia tos pačios Python aplinkos kaip 7 skyriuje ir `Rscript` su „Cairo“ palaikymu. Failai `A25_7_vertinimas.py` ir `A25_6_klasterizavimas.R` turi būti tame pačiame aplanke kaip grafiko skriptas.

```bash
python A25_7_ARI_grafikas.py \
  A25_clean.csv \
  rezultatai_7/A25_7_imciu_poros.csv \
  rezultatai_7/A25_7_ARI_pakankamumas.png
```

Grafikas išsaugomas faile `rezultatai_7/A25_7_ARI_pakankamumas.png`.

## 8 skyrius. Klasterių profiliai

Reikia Python 3. Papildomų bibliotekų diegti nereikia.

Paleiskite komandą iš repozitorijos aplanko po 6 skyriaus. Skriptui reikia `A25_clean.csv` ir 6 skyriuje sugeneruoto `rezultatai_6/A25_6_priskyrimai.csv`.

```bash
python3 A25_8_profiliai.py \
  A25_clean.csv \
  rezultatai_6/A25_6_priskyrimai.csv \
  rezultatai_8/A25_8_profiliai.csv
```

Skriptas apskaičiuoja PCA ir „k-means“ klasterių dydžius, defektų klasių pasiskirstymą bei požymių medianas ir kvartilius. Rezultatus išsaugo faile `rezultatai_8/A25_8_profiliai.csv`.
