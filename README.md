# Paikallinen puhe-RAG-demo

Suomenkielinen puhedemo, jossa äänentunnistus, PDF-haku, vastausten generointi
ja puhesynteesi tehdään omalla koneella.

Normaali ajo on pakotettu offline-tilaan. Mallien lataaminen on erillinen,
käyttajän tietoisesti käynnistämä vaihe. `tietokanta/`- ja `models/`-kansioita
ei lisätä Gitiin.

## Asennus

Luo virtuaaliymparistö ja asenna riippuvuudet:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Lataa julkiset mallit kerran verkkoyhteyden aikana:

```powershell
python local_rag_puhedemo.py --download-models
```

Latauksen kokonaiskoko on noin 5 GB. Upotusmallista ladataan vain ohjelman
tarvitsema PyTorch-versio; vaihtoehtoisia ONNX-, OpenVINO- ja TensorFlow-
mallipainoja ei ladata.

Lisää omat PDF-tiedostot paikalliseen `tietokanta`-kansioon. Kansio luodaan
automaattisesti ensimmäisessä ajossa, eika sitä seurata Gitissa.

Käynnistä tämän jälkeen offline-ajo:

```powershell
python local_rag_puhedemo.py
```

Paina välilyöntiä kerran aloittaaksesi äänityksen. Paina välilyöntiä uudelleen
lopettaaksesi äänityksen ja käynnistääksesi kysymyksen käsittelyn. Lopeta
ohjelma painamalla `q`.

Litteroinnit ja haetut tekstikatkelmat eivat tulostu oletuksena.
Paikalliseen vianhakuun ne saa nakyviin komennolla:

```powershell
python local_rag_puhedemo.py --debug
```

Mallien ja PDF-kansion sijainnin voi vaihtaa ilman lähdekoodin muokkaamista:

```powershell
$env:LOCAL_RAG_MODELS_DIR = "D:\mallit"
$env:LOCAL_RAG_PDF_DIR = "D:\asiakirjat"
python local_rag_puhedemo.py
```

Ohjelmallinen verkkosuoja estaa tavalliset Pythonin verkkoyhteydet normaalissa
ajossa. Tiukassa ymparistossa kannattaa lisaksi estaa Pythonin lahteva liikenne
kayttojarjestelman palomuurilla.
