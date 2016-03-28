__author__ = 'Marek'
from qbrain.go.gtp import GoTextPipe


class Go():
    alpha_positions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
    alpha_values = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'J': 8, 'K': 9, 'L': 10, 'M': 11,
                    'N': 12, 'O': 13, 'P': 14, 'Q': 15, 'R': 16, 'S': 17, 'T': 18}

    black_str = 'black'
    white_str = 'white'
    pass_str = 'pass'

    board_size = 19

    empty_field = 0
    black_field = 1
    white_field = -1

    empty_field_char = ' '
    black_field_char = '.'
    white_field_char = 'O'

    def __init__(self):
        self.go = GoTextPipe(board_size=Go.board_size)
        self.next = Go.black_str
        self.last_has_passed = False
        self.is_finished = False
        self.winner = None
        self.score = None

    def switch_next(self):
        if self.next == Go.black_str:
            self.next = Go.white_str
        else:
            self.next = Go.black_str

    def expert_move(self):
        if not self.is_finished:
            genmove = self.go.genmove(self.next).lower()
            self.switch_next()

            if genmove == Go.pass_str:
                if self.last_has_passed:
                    self.finish_game()
                else:
                    self.last_has_passed = True
                return None, Go.pass_str
            else:
                self.last_has_passed = False
                x = Go.alpha_values[genmove[0].upper()]
                y = int(genmove[1:]) - 1
                return (x, y), None

    def move_pass(self):
        if not self.is_finished:
            self.go.play(self.next, Go.pass_str)
            self.switch_next()
            if self.last_has_passed:
                self.finish_game()
            else:
                self.last_has_passed = True

    def move(self, x, y):
        if not self.is_finished:
            position = Go.alpha_positions[x] + str((y + 1))
            self.go.play(self.next, position)
            self.last_has_passed = False
            self.switch_next()

    def finish_game(self):
        if not self.is_finished:
            self.is_finished = True
            final_score = self.go.final_score()
            winner = final_score[1]
            score = float(final_score[2:])

            if winner.upper() == 'W':
                self.winner = Go.white_str
            else:
                self.winner = Go.black_str

            self.score = score

    def get_black_stones(self):
        stones_str = self.go.list_stones(Go.black_str)
        if stones_str == '':
            stones = []
        else:
            stones = stones_str.split(' ')
        return stones

    def get_white_stones(self):
        stones_str = self.go.list_stones(Go.white_str)
        if stones_str == '':
            stones = []
        else:
            stones = stones_str.split(' ')
        return stones

    def get_field(self):
        field = [None] * Go.board_size
        for i in range(Go.board_size):
            field[i] = [Go.empty_field] * Go.board_size

        for stone in self.get_black_stones():
            x = Go.alpha_values[stone[0].upper()]
            y = int(stone[1:]) - 1
            field[y][x] = Go.black_field

        for stone in self.get_white_stones():
            x = Go.alpha_values[stone[0].upper()]
            y = int(stone[1:]) - 1
            field[y][x] = Go.white_field

        return field

    def get_field_as_str(self):
        field_str = ''
        field = self.get_field()
        for y in range(len(field)):
            row = field[y]
            for x in range(len(row)):
                if row[x] == Go.empty_field:
                    char = Go.empty_field_char
                elif row[x] == Go.black_field:
                    char = Go.black_field_char
                else:
                    char = Go.white_field_char

                field_str += char + ' '
            field_str += '\n'
        return field_str

    def legal_white_moves(self):
        legal_moves_str = self.go.legal_moves(Go.white_str)
        legal_moves = legal_moves_str.split(' ')
        return legal_moves

    def legal_black_moves(self):
        legal_moves_str = self.go.legal_moves(Go.black_str)
        legal_moves = legal_moves_str.split(' ')
        return legal_moves

    def show_board(self):
        return self.go.showboard()