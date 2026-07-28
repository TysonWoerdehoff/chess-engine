import chess
import chess.polyglot
import torch
from search import best_move
from model import ChessNet



def play():
    model = ChessNet()
    model.load_state_dict(torch.load("models/chess_net.pt"))
    model.eval()
    try:
        book = chess.polyglot.open_reader("data/gm2001.bin")
    except FileNotFoundError:
        print("No opening book found, playing without one")
        book = None
    board = chess.Board()
    while(not board.is_game_over()):
        print(board)
        move_in = input("Enter a move: ")
        try:
            board.push_san(move_in)
        except ValueError:
            print("Illegal move, try again")
            continue
        if board.is_game_over(): break
        mv = best_move(board, 3, model, book)
        print("Engine plays:", board.san(mv))
        board.push(mv)
    print(board.result())


if __name__ == "__main__":
    play()