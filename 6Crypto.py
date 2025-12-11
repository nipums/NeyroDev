import chess.pgn

""" load the first game from a PGN file """
with open("moves.pgn") as pgn:
    first_game = chess.pgn.read_game(pgn)

if not first_game:
    raise ValueError("No game found")