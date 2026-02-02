Intelligent Idea Analysis Engine (IIAE) 

Dette prosjektet presenterer en komplett data-pipeline og analyseplattform for intelligent prosessering av brukergenererte ideer. Systemet går utover tradisjonell nøkkelordsmatching ved å bruke Computational Intelligence (CI) for å forstå den semantiske meningen bak tekst.

Prosjektmål
Hovedmålet med IIAE er å automatisere håndteringen av store mengder ideer ved å:


Kategorisere bidrag automatisk inn i relevante temaer som Helse, Verdensrommet og IT-sikkerhet.


Detektere duplikater basert på betydning, ikke bare ordvalg.


Validere originalitet ved å sjekke nye bidrag mot en eksisterende database med 100 000 ideer.


Visualisere data i et semantisk landskap for å identifisere trender og klynger.

 Teknologi
Prosjektet er bygget på en moderne teknisk stack:


Backend: Python med Microsoft SQL Server for lagring av 100 000 unike ideer.


AI Model: Sentence-BERT (SBERT) for generering av høy-dimensjonale vektorer (embeddings).


Visualisering: PCA og t-SNE for reduksjon av kompleksitet til 2D-kart.

Web Interface: Streamlit for interaktiv demonstrasjon.

Resultater
Ved bruk av SBERT har jeg lykkes med å kartlegge 100 000 ideer i et semantisk univers. Visualiseringen (se bilde under) viser tydelige klynger innenfor de åtte hovedkategoriene, noe som bekrefter modellens evne til å forstå konseptuelle forskjeller.

(Her kan du sette inn bildet av grafen din i GitHub)

Slik bruker du applikasjonen
Åpne live-demoen via [DIN_STREAMLIT_LINK].

Naviger til feltet "Submit Idea".

Skriv inn en idé (f.eks. om romfart eller helseteknologi).

Se hvordan systemet umiddelbart kategoriserer ideen og validerer den mot databasen.
