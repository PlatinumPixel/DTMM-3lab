# DTMM 3-ias laboratorinis darbas

A25 duomenų dimensijos mažinimas, klasterizavimas ir klasterių kokybės bei stabilumo vertinimas.

## Reikalavimai

- Python 3 su `numpy` ir `pillow` bibliotekomis.
- R su terminale pasiekiama `Rscript` komanda. Papildomų R paketų nereikia.
- `A25_clean.csv` ir `A25_normuotas.csv` failai tame pačiame aplanke kaip skriptai. Juos reikia pridėti atskirai.

Atidarykite terminalą repozitorijos aplanke. Jei Python aplinka dar nesukurta, paleiskite šias komandas (bash):

python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pillow

## 5 skyrius. Dimensijos mažinimas

Apskaičiuoja PCA bei t-SNE koordinates ir sukuria projekcijų grafiką. Rezultatai išsaugomi aplanke `rezultatai_5` (bash).

python A25_5_dimensijos_mazinimas.py \
  A25_normuotas.csv rezultatai_5

## 6 skyrius. Klasterizavimas

Paleiskite po 5 skyriaus. Skriptas taiko „k-means“, Ward ir DBSCAN metodus pradinei bei sumažintoms erdvėms. Klasterių priskyrimai, suvestinės ir grafikas išsaugomi aplanke `rezultatai_6` (bash).

Rscript A25_6_klasterizavimas.R \
  A25_normuotas.csv \
  rezultatai_5/A25_5_koordinates.csv \
  rezultatai_6


## 7 skyrius. Vertinimas

Paleiskite po 5 ir 6 skyrių. Skriptas apskaičiuoja klasterių kokybės matus ir įvertina stabilumą dešimtyje kartotinių imčių. Rezultatų lentelės išsaugomos aplanke `rezultatai_7` (bash).

python A25_7_vertinimas.py \
  A25_clean.csv \
  A25_normuotas.csv \
  rezultatai_5/A25_5_koordinates.csv \
  rezultatai_6/A25_6_priskyrimai.csv \
  A25_5_dimensijos_mazinimas.py \
  A25_6_klasterizavimas.R \
  rezultatai_7
