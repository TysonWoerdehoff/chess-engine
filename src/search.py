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

PASSED_PAWN_BONUS = 20
KING_ACTIVITY_WEIGHT = 8
ENDGAME_THRESHOLD = 1800


def material_balance(board: chess.Board) -> float:
    total = 0
    
    for piece in board.piece_map().values():
        if (piece.color): total += PIECE_VALUES[piece.piece_type]
        else: total -= PIECE_VALUES[piece.piece_type]
    total += passed_pawn_bonus(board)
    total += king_activity_bonus(board)
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
    for move in order_moves(board):
        board.push(move)
        score = -negamax(board, depth - 1, model, ply + 1)
        board.pop()
        best = max(best, score)
    return best

def move_score(board: chess.Board, move: chess.Move) -> int:
    if not board.is_capture(move):
        return 0
    victim = board.piece_at(move.to_square)
    attacker = board.piece_at(move.from_square)
    victim_value = PIECE_VALUES[victim.piece_type] if victim else PIECE_VALUES[chess.PAWN]
    return 10 * victim_value - PIECE_VALUES[attacker.piece_type]


def order_moves(board: chess.Board) -> list:
    return sorted(board.legal_moves, key=lambda m: move_score(board, m), reverse=True)

def best_move(board, depth, model, book=None) -> chess.Move:
    if book is not None:
        try: 
            return book.weighted_choice(board).move
        except IndexError:
            pass
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

def quiescence(board: chess.Board, model: ChessNet, ply: int = 0, qply: int = 0, max_qdepth: int = 4) -> float:
    if board.is_checkmate(): return -(MATE_SCORE - ply) 
    best = evaluate(board, model)
    if (max_qdepth <= qply): return best
    for move in order_moves(board):
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
    for move in order_moves(board):
        board.push(move)
        score = - alphabeta(board, depth - 1, -beta, -alpha, model, ply+1)
        board.pop()
        if (score >= beta): return beta
        if (score > alpha): alpha = score
    return alpha

def is_passed(board: chess.Board, sq: int, color: bool) -> bool:
    file = sq % 8
    rank = sq // 8
    for ep in board.pieces(chess.PAWN, not color):
        if abs((ep % 8) - file) > 1:
            continue
        ep_rank = ep // 8
        if color == chess.WHITE and ep_rank > rank:
            return False
        if color == chess.BLACK and ep_rank < rank:
            return False
    return True


def passed_pawn_bonus(board: chess.Board) -> int:
    total = 0
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        if is_passed(board, sq, chess.WHITE):
            total += PASSED_PAWN_BONUS * (sq // 8)
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        if is_passed(board, sq, chess.BLACK):
            total -= PASSED_PAWN_BONUS * (7 - (sq // 8))
    return total


def non_pawn_material(board: chess.Board) -> int:
    total = 0
    for piece in board.piece_map().values():
        if piece.piece_type not in (chess.PAWN, chess.KING):
            total += PIECE_VALUES[piece.piece_type]
    return total


def centre_distance(sq: int) -> float:
    return abs((sq % 8) - 3.5) + abs((sq // 8) - 3.5)


def king_activity_bonus(board: chess.Board) -> float:
    if non_pawn_material(board) > ENDGAME_THRESHOLD:
        return 0
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return 0
    return KING_ACTIVITY_WEIGHT * (centre_distance(bk) - centre_distance(wk))
    
    
if __name__ == "__main__":
    model = ChessNet()
    model.load_state_dict(torch.load("models/chess_net.pt"))
    model.eval()

    b1 = chess.Board("4k3/8/8/3P4/8/8/8/4K3 w - - 0 1")
    b2 = chess.Board("4k3/8/8/3P4/8/8/8/4K3 b - - 0 1")
    print(material_balance(b1), material_balance(b2))       