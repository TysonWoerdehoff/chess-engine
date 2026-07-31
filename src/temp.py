import chess
import torch
import time
import random
from model import ChessNet
import search
from search import material_balance, alphabeta, order_moves

model = ChessNet()
model.load_state_dict(torch.load("models/chess_net.pt"))
model.eval()

DEPTH = 2
GAMES = 20
MAX_PLIES = 250
RANDOM_OPENING_PLIES = 4

random.seed(42)


def pick(board, depth, use_network):
    """Root search. If use_network is False, evaluation is material+heuristics only."""
    original = search.evaluate
    if not use_network:
        search.evaluate = lambda b, m: material_balance(b)
    try:
        best_score = -float("inf")
        best_mv = None
        for move in order_moves(board):
            board.push(move)
            score = -alphabeta(board, depth - 1, -float("inf"), -best_score, model, 1)
            board.pop()
            if score > best_score:
                best_score = score
                best_mv = move
        return best_mv
    finally:
        search.evaluate = original


nn_wins = mat_wins = draws = 0
t_start = time.time()

for game in range(GAMES):
    board = chess.Board()
    for _ in range(RANDOM_OPENING_PLIES):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))
    opening_fen = board.fen()

    nn_is_white = game % 2 == 0
    plies = 0
    while not board.is_game_over() and plies < MAX_PLIES:
        board.push(pick(board, DEPTH, board.turn == nn_is_white))
        plies += 1

    result = board.result()
    if plies >= MAX_PLIES or result == "1/2-1/2":
        draws += 1
        outcome = "draw"
    elif (result == "1-0") == nn_is_white:
        nn_wins += 1
        outcome = "NN"
    else:
        mat_wins += 1
        outcome = "material"

    print(f"game {game:2}  nn={'W' if nn_is_white else 'B'}  {result:8} {plies:3} plies  winner: {outcome}")

print(f"\nNN: {nn_wins}   material-only: {mat_wins}   draws: {draws}")
print(f"({time.time()-t_start:.0f}s total)")