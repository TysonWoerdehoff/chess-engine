import chess
import torch

def board_to_tensor(board: chess.Board) -> torch.Tensor: 
    t = torch.zeros(17,8,8)

    for i in range(64):
        piece = board.piece_at(i)
        if piece is None:
            continue
        rank = i // 8
        file = i % 8
        plane = (piece.piece_type - 1)  + (0 if(piece.color) else 6)
        t[plane,rank, file] = 1
    
    t[12,:,:] = 1.0 if (board.turn) else 0.0
    t[13,:,:] = 1.0 if (board.has_kingside_castling_rights(chess.WHITE)) else 0.0
    t[14,:,:] = 1.0 if (board.has_queenside_castling_rights(chess.WHITE)) else 0.0
    t[15,:,:] = 1.0 if (board.has_kingside_castling_rights(chess.BLACK)) else 0.0
    t[16,:,:] = 1.0 if (board.has_queenside_castling_rights(chess.BLACK)) else 0.0
    return t
    
if __name__ == "__main__": 
    board = chess.Board()
    t = board_to_tensor(board)
    print("shape:", t.shape)
    print("dtype:", t.dtype)
    print("white pawns (plane 0):\n", t[0])
    print("total pieces encoded:", t[:12].sum().item())
