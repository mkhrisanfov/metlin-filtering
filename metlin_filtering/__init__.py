from pathlib import Path

SEED = 44
RADIUS = 3
BASE_DIR = Path(__file__).resolve(strict=True).parent.parent
OUTPUT_DIR = BASE_DIR / "data/output"
PROCESSED_DIR = BASE_DIR / "data/processed"
MODEL_NAMES = [
    "CNN",
    "FCD",
    "FCFP",
    "GNN",
    "CB"
]


if __name__ == "__main__":
    from metlin_filtering.training import main
    main(filename=BASE_DIR / "data" / "input" / "SMRT_dataset.csv",
         train=True,
         predict=True)
