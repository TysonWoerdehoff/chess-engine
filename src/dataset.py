import chess
import pandas as pd
import torch
from torch.utils.data import Dataset

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


class ChessDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        board = chess.Board(row["FEN"])
        x = board_to_tensor(board)
        
        value = row["eval_cp"]
        if (not board.turn): 
            value = -value
        value /= EVAL_CLAMP
        
        y = torch.tensor([value], dtype=torch.float32)
        return x, y
    
    
    
if __name__ == "__main__":
    df = load_dataframe("data/chessData.csv", n_rows = 100_000)
    ds = ChessDataset(df)
    print("len:", len(ds))
    x, y = ds[0]
    print("x shape:", x.shape, "x dtype:", x.dtype)
    print("y:", y, "y shape:", y.shape)
    
    x, y = ds[1]
    print("second sample y:", y)