import json
import os
import urllib.request

import pandas as pd

URL = "https://raw.githubusercontent.com/risan/quran-json/main/dist/quran_en.json"
DATA_DIR = os.environ.get("NOOR_DATA_DIR", "/config")
OUT = os.path.join(DATA_DIR, "quran-dataset.csv")

os.makedirs(DATA_DIR, exist_ok=True)
with urllib.request.urlopen(URL) as r:
    data = json.load(r)

rows = []
for surah in data:
    for verse in surah.get("verses", []):
        rows.append({
            "surah_no": surah["id"],
            "ayah_no_surah": verse["id"],
            "ayah_en": verse.get("translation", ""),
            "ayah_ar": verse.get("text", ""),
        })

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"Wrote {len(rows)} verses to {OUT}")
