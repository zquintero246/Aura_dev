import pandas as pd

ds = pd.read_csv(r"C:\Users\Zabdiel Julian\Downloads\IngenieriaSoftwareII\Niggahush_dev\NiggahushLM\test\data\poems_content.csv"
                 )
ds = ds[['content']]

ds['content'] = ds['content'].str.replace(r'\n', ' ', regex=True).str.strip()

ds['content'] = ds['content'].str.lower()

print(ds.head())
print(ds.columns)

corpus = " ".join(ds["content"].astype(str).tolist())

with open("corpus_poems.txt", "w", encoding="utf-8") as f:
    f.write(corpus)

print("Corpus creado con longitud:", len(corpus), "caracteres")