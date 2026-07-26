import chess
import torch
import time

from encode import board_to_tensor
from model import ChessNet

def evaluate(board: chess.Board, model: ChessNet) -> float:
    x = board_to_tensor(board)
    x = x.unsqueeze(0)
    with torch.no_grad():
        return model(x).item() + material_balance(board)


    
MATE_SCORE = 1000.0

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

def material_balance(board: chess.Board) -> float:
    total = 0
    for piece in board.piece_map().values():
        if (piece.color): total += PIECE_VALUES[piece.piece_type]
        else: total -= PIECE_VALUES[piece.piece_type]
    if (not board.turn): total = -total
    return total/1000


def negamax(board: chess.Board, depth: int, model: ChessNet, ply: int = 0) -> float:
    if board.is_checkmate():
        return -(MATE_SCORE - ply)
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if depth == 0:
        return quiescence(board, model, ply, 0, 2)

    best = -float("inf")
    for move in board.legal_moves:
        board.push(move)
        score = -negamax(board, depth - 1, model, ply + 1)
        board.pop()
        best = max(best, score)
    return best

def best_move(board, depth, model) -> chess.Move:
    best_score = -float("inf")
    best_mv = None
    for move in board.legal_moves:
        board.push(move)
        score = -alphabeta(board, depth - 1, -float("inf"), -best_score, model, 1)
        board.pop()
        if score > best_score:
            best_score = score
            best_mv = move
    return best_mv

def quiescence(board: chess.Board, model: ChessNet, ply: int = 0, qply: int = 0, max_qdepth: int = 4) -> float:
    if board.is_checkmate(): return -(MATE_SCORE - ply) 
    best = evaluate(board, model)
    if (max_qdepth <= qply): return best
    for move in board.legal_moves:
        if(board.is_capture(move)):
            board.push(move)
            score = -quiescence(board, model, ply + 1, qply + 1, max_qdepth)
            board.pop()
            best = max(best, score)
    return best

def alphabeta(board, depth, alpha, beta, model, ply=0) -> float:
    if board.is_checkmate():
        return -(MATE_SCORE - ply)
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if depth == 0:
        return quiescence(board, model, ply, 0, 2)
    for move in board.legal_moves:
        board.push(move)
        score = - alphabeta(board, depth - 1, -beta, -alpha, model, ply+1)
        board.pop()
        if (score >= beta): return beta
        if (score > alpha): alpha = score
    return alpha
    
    
    
if __name__ == "__main__":
    model = ChessNet()
    model.load_state_dict(torch.load("models/chess_net.pt"))
    model.eval()

    b = chess.Board("r1bqkb1r/pppppppp/2n2n2/8/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 0 1")
    
    t0 = time.time()
    s1 = negamax(b, 3, model)
    print(f"negamax:   {s1:.6f} in {time.time()-t0:.1f}s")
    
    t0 = time.time()
    s2 = alphabeta(b, 3, -float("inf"), float("inf"), model)
    print(f"alphabeta: {s2:.6f} in {time.time()-t0:.1f}s")