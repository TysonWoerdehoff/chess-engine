import chess
import torch
import time
from search import best_move
from model import ChessNet

model = ChessNet()
model.load_state_dict(torch.load("models/chess_net.pt"))
model.eval()

POSITIONS = [
    ("opening",    chess.Board()),
    ("developed",  chess.Board("r1bqkb1r/pppppppp/2n2n2/8/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 0 1")),
    ("tactical",   chess.Board("r1bqkb1r/pppp1ppp/4p3/1B1P4/3Q4/2N5/PPP2PPP/R1B1K1NR b - - 0 7")),
    ("endgame",    chess.Board("8/5p2/6kp/7P/2P2P2/1P1N4/R7/4K3 w - - 0 1")),
]

DEPTHS = [1, 2, 3, 4]

print(f"{'position':12} {'depth':>5} {'move':>8} {'seconds':>9}")
print("-" * 38)

for name, board in POSITIONS:
    for depth in DEPTHS:
        t0 = time.time()
        mv = best_move(board, depth, model)
        elapsed = time.time() - t0
        print(f"{name:12} {depth:>5} {board.san(mv):>8} {elapsed:>9.2f}")
    print()