from fastapi import FastAPI
from pathlib import Path
import json

app = FastAPI()


def load_top_results():
    root = Path(__file__).resolve().parents[2]
    fp = root / "outputs" / "top_results.json"
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


@app.get("/top-results")
def top_results():
    return {"results": load_top_results()}
