import chess
import chess.svg
from IPython.display import SVG, display
import random
import time

piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
        }

piece_values_with_position = {
        chess.PAWN: [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [10, 10, 20, 30, 30, 20, 10, 10],
            [5, 5, 10, 25, 25, 10, 5, 5],
            [0, 0, 0, 20, 20, 0, 0, 0],
            [5, -5, -10, 0, 0, -10, -5, 5],
            [5, 10, 10, -20, -20, 10, 10, 5],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ],
        chess.KNIGHT: [
            [-50, -40, -30, -30, -30, -30, -40, -50],
            [-40, -20, 0, 0, 0, 0, -20, -40],
            [-30, 0, 10, 15, 15, 10, 0, -30],
            [-30, 5, 15, 20, 20, 15, 5, -30],
            [-30, 0, 15, 20, 20, 15, 0, -30],
            [-30, 5, 10, 15, 15, 10, 5, -30],
            [-40, -20, 0, 5, 5, 0, -20, -40],
            [-50, -40, -30, -30, -30, -30, -40, -50]
        ],
        chess.BISHOP: [
            [-20, -10, -10, -10, -10, -10, -10, -20],
            [-10, 0, 0, 0, 0, 0, 0, -10],
            [-10, 0, 5, 10, 10, 5, 0, -10],
            [-10, 5, 5, 10, 10, 5, 5, -10],
            [-10, 0, 10, 10, 10, 10, 0, -10],
            [-10, 10, 10, 10, 10, 10, 10, -10],
            [-10, 5, 0, 0, 0, 0, 5, -10],
            [-20, -10, -10, -10, -10, -10, -10, -20]
        ],
        chess.ROOK: [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [5, 10, 10, 10, 10, 10, 10, 5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [0, 0, 0, 5, 5, 0, 0, 0]
        ],
        chess.QUEEN: [
            [-20, -10, -10, -5, -5, -10, -10, -20],
            [-10, 0, 0, 0, 0, 0, 0, -10],
            [-10, 0, 5, 5, 5, 5, 0, -10],
            [-5, 0, 5, 5, 5, 5, 0, -5],
            [0, 0, 5, 5, 5, 5, 0, -5],
            [-10, 5, 5, 5, 5, 5, 0, -10],
            [-10, 0, 5, 0, 0, 0, 0, -10],
            [-20, -10, -10, -5, -5, -10, -10, -20]
        ],
        chess.KING: [
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-20, -30, -30, -40, -40, -30, -30, -20],
            [-10, -20, -20, -20, -20, -20, -20, -10],
            [20, 20, 0, 0, 0, 0, 20, 20],
            [20, 30, 10, 0, 0, 10, 30, 20]
        ]
    }

class MyChessBot:
    def __init__(self) :
        self.piece_values = piece_values
        self.piece_values_with_position = piece_values_with_position

    def random_move(self, board):
        moves = list(board.legal_moves)
        move = random.choice(moves)
        return move
    
    def old_minimax(self, board, depth, alpha, beta, white_to_play):
        if depth == 0:
            return self.evaluate_board(board), None
        if white_to_play:
            best_score = -9999
            best_move = None

            for move in board.legal_moves:
                board.push(move)
                score, _ = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break
            return best_score, best_move
        else:
            best_score = 9999
            best_move = None
            for move in board.legal_moves:
                board.push(move)
                score, _ = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, best_score)
                if alpha >= beta:
                    break
            return best_score, best_move

    def minimax(self, board, depth, alpha, beta, white_to_play, memo={}):
        if depth == 0:
            return self.evaluate_board(board), None
        board_key = board.fen()  # Unique identifier for the board position
        if board_key in memo: # memo stops the ai from reavaluating already evaluated nodes
            return memo[board_key]

        if white_to_play:
            best_score = -9999
            best_move = None

            for move in board.legal_moves:
                board.push(move)
                score, _ = self.minimax(board, depth - 1, alpha, beta, False, memo)
                board.pop()
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break
            memo[board_key] = best_score, best_move
            return best_score, best_move
        else:
            best_score = 9999
            best_move = None
            for move in board.legal_moves:
                board.push(move)
                score, _ = self.minimax(board, depth - 1, alpha, beta, True, memo)
                board.pop()
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, best_score)
                if alpha >= beta:
                    break
            memo[board_key] = best_score, best_move
            return best_score, best_move
        
    def best_move_using_minimax(self, board, depth):
        white_to_play = board.turn
        best_score = -99999 if white_to_play else 99999
        best_move = self.random_move(board)
        for move in board.legal_moves:
            board.push(move)
            score, _ = self.minimax(board, depth - 1, -99999, 99999, not white_to_play)
            board.pop()
            if (white_to_play and score > best_score) or (not white_to_play and score < best_score):
                best_score = score
                best_move = move
        return best_move

    def get_piece_value(self, piece, x, y):
        if piece.piece_type == chess.PAWN:
            return 100 + piece_values_with_position[piece.piece_type][x][y]
        elif piece.piece_type == chess.KNIGHT:
            return 320 + piece_values_with_position[piece.piece_type][x][y]
        elif piece.piece_type == chess.BISHOP:
            return 330 + piece_values_with_position[piece.piece_type][x][y]
        elif piece.piece_type == chess.ROOK:
            return 500 + piece_values_with_position[piece.piece_type][x][y]
        elif piece.piece_type == chess.QUEEN:
            return 900 + piece_values_with_position[piece.piece_type][x][y]
        elif piece.piece_type == chess.KING:
            return 20000 + piece_values_with_position[piece.piece_type][x][y]
        else:
            return 0

    def evaluate_board(self, board):
        score = 0
        for i in range(8):
            for j in range(8):
                piece = board.piece_at(chess.square(i, j))
                if piece is not None:
                    if piece.color == chess.WHITE:
                        score += self.get_piece_value(piece, i, j)
                    else:
                        score -= self.get_piece_value(piece, i, j)
        return score

    def print_board(self, board,size=(150,100)):
        print(board)

    def play_game(self):
        board = chess.Board()
        self.print_board(board)
        move_times = []

        while not board.is_game_over():
            #if board.turn == chess.WHITE:
            #    user_move = input("Your move (in algebraic notation, e.g., e2e4): ")
            #    if chess.Move.from_uci(user_move) in board.legal_moves:
            #        board.push_uci(user_move)
            #    else:
            #        print("Invalid move. Try again.")
            #        continue
            #else:
            start_time = time.time()
            ai_move = self.best_move_using_minimax(board, depth=5)
            board.push(ai_move)
            print(ai_move)
            move_time = time.time() - start_time
            move_times.append(move_time)
            print(f"Time used to find move: {move_time} seconds")

            print("---------------")
            self.print_board(board,size=(150,100))

        print("Game Over")
        print("Result: ", board.result())
        print(f"Avarage time spend per move: {sum(move_times)/len(move_times)} seconds")

chess_game = MyChessBot()
chess_game.play_game()