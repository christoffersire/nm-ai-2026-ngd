# CLIP Re-annotation — kjør på GPU-VM

## Hva dette gjør
Re-klassifiserer alle 22 731 annotations ved å matche hvert bbox-crop
mot CLIP-embeddings bygget fra referansebildene i NM_NGD_product_images/.

Fikser:
- Feilmerkede annotations (feil kategori_id)
- Cat#355 (unknown_product) — alle 422 flyttes til riktig kategori
- Generelt støy i annotasjonene

## Installasjon (én gang)
```bash
pip install open-clip-torch torch Pillow
```

## Kjør
```bash
# Fra prosjektrotmappen nm-ai-2026-ngd/
python3 data/clip_reannotate.py
```

Med større og bedre modell (anbefalt hvis dere har god GPU):
```bash
python3 data/clip_reannotate.py --model ViT-L-14 --pretrain openai --threshold 0.80
```

Dry-run først for å se hva som vil skje:
```bash
python3 data/clip_reannotate.py --dry-run
```

## Parametre
| Parameter | Standard | Beskrivelse |
|-----------|----------|-------------|
| `--model` | ViT-B-32 | CLIP-modell. ViT-L-14 er bedre men tregere |
| `--threshold` | 0.77 | Cosine-likhet for auto-reassignment (0-1) |
| `--batch-size` | 64 | Bilder per batch (øk for raskere kjøring) |
| `--no-cache` | False | Tving reberegning av category embeddings |

## Output
| Fil | Beskrivelse |
|-----|-------------|
| `~/Downloads/train/annotations_clip.json` | **Bruk denne til trening** |
| `data/audit_out/clip_changes.csv` | Hva som ble endret + confidence |
| `data/audit_out/clip_unmatched.csv` | Lav-konfidens crops (review disse) |
| `data/audit_out/clip_embeddings.pt` | Cache (raskere neste kjøring) |

## Estimert kjøretid
| Hardware | ViT-B-32 | ViT-L-14 |
|----------|----------|----------|
| GPU (T4) | ~8 min | ~20 min |
| GPU (A100) | ~3 min | ~8 min |
| CPU | ~4 timer | ~12 timer |

## Etter kjøring
Bytt `annotations.json` → `annotations_clip.json` i treningspipelinen:
```python
# I prepare_yolo.py / prepare_classifier_data.py:
ANNOTATIONS_PATH = Path.home() / "Downloads" / "train" / "annotations_clip.json"
```
