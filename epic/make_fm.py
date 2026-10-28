"""metadata JSON + file_mappings для csRNA EPIC (по одному треку на вид).

    python epic/make_fm.py [ASSEMBLY ...]      # по умолчанию все виды, у которых уже есть bw

Выход: epic_train/A/metadata/A.json, epic_train/A/file_mappings_A.csv (пути относительные).
Имя file_mappings уникально для вида: кэш описаний AIRI (descriptions/) ключуется по имени
файла и его размеру, и одинаковое имя у разных видов может подменить описание.

JSON превращается в текст как "key is value." по каждому полю (FixedDescriptionsMixin),
поэтому ключи - обычные слова. ЭТОТ ЖЕ JSON подаём в predict.py на инференсе.
"""
import json, os, sys
import pandas as pd

H = os.environ["GENALM_HOME"]
OUT = os.path.join(H, "epic_train")
SPECIES = {  # сборка: (вид, ткань)
    "ASM119413v2": ("Octopus bimaculoides", "kidney"),
    "sSquSuc5.hap2": ("Squalus suckleyi", "blood"),
    "Ofas_2.0": ("Oncopeltus fasciatus", "embryo"),
    "cgigas_uk_roslin_v1": ("Magallana gigas", "heart"),
    "ilPloInte3.2": ("Plodia interpunctella", "embryo"),
    "jaNemVect1.1": ("Nematostella vectensis", "mixed tissues"),
}
ASSAY_TEXT = ("5' ends of capped small RNAs, marking RNA polymerase II transcription "
              "initiation sites, strand-specific")

for A in sys.argv[1:] or SPECIES:
    d = os.path.join(OUT, A)
    bw = {s: f"bw/{A}.pool.{s}.bw" for s in ("forward", "reverse")}
    if not all(os.path.exists(os.path.join(d, p)) for p in bw.values()):
        print(f"{A}: нет bigWig, пропускаю")
        continue
    species, tissue = SPECIES[A]
    meta = {"assay": "csRNA-seq", "species": species, "tissue": tissue,
            "description": ASSAY_TEXT, "genome": A}
    os.makedirs(os.path.join(d, "metadata"), exist_ok=True)
    json.dump(meta, open(os.path.join(d, "metadata", f"{A}.json"), "w"), indent=1)
    pd.DataFrame([{
        "id": f"csRNA_{A}", "metadata": f"./metadata/{A}.json", "assay": "csRNA-seq",
        "genome": A, "forward_bw": f"./{bw['forward']}", "reverse_bw": f"./{bw['reverse']}",
        "dataset_description": f"EPIC csRNA {A}", "processed": "kondratiuk",
    }]).to_csv(os.path.join(d, f"file_mappings_{A}.csv"), index=False)
    text = " ".join(f"{k.replace('_', ' ')} is {v}." for k, v in meta.items())
    print(f"{A}: ок | описание: {text}")
