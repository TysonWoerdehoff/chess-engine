import chess
import pandas as pd
import torch

from encode import board_to_tensor

EVAL_CLAMP = 1000.0
MATE_SCORE = 10000.0

def parse_eval(value: str) -> float:
    value = value.strip()
    if (value.startswith("#")):
        if (value[1] == "-"):
            return -MATE_SCORE
        return MATE_SCORE
    return float(value)

def load_dataframe(csv_path: str, n_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, nrows= n_rows)
    df["eval_cp"] = df["Evaluation"].apply(parse_eval)
    df["eval_cp"] = df["eval_cp"].clip(-EVAL_CLAMP, EVAL_CLAMP)
    return df

if __name__ == "__main__":
    df = load_dataframe("data/chessData.csv")
    print(df.describe())