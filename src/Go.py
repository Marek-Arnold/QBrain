__author__ = 'Marek'
import gtp


class Go():
    alpha_positions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
    alpha_values = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                    'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19}

    black_str = 'black'
    white_str = 'white'
    pass_str = 'pass'

    board_size = 9
    empty_field = 0
    black_field = 1
    white_field = -1

    def __init__(self):
        self.go = gtp.GoTextPipe()
        self.next = Go.black_str
        self.last_has_passed = False
        self.is_finished = False

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
                    self.is_finished = True
                else:
                    self.last_has_passed = True
            else:
                self.last_has_passed = False

    def move_pass(self):
        if not self.is_finished:
            self.go.play(self.next, Go.pass_str)
            self.switch_next()
            if self.last_has_passed:
                self.is_finished = True
            else:
                self.last_has_passed = True

    def move(self, x, y):
        if not self.is_finished:
            position = Go.alpha_positions[x] + str((y + 1))
            self.go.play(self.next, position)
            self.switch_next()

    def get_black_stones(self):
        stones_str = self.go.list_stones(Go.black_str)
        stones = stones_str.split(' ')
        return stones

    def get_white_stones(self):
        stones_str = self.go.list_stones(Go.white_str)
        stones = stones_str.split(' ')
        return stones

    def get_field(self):
        field = [None] * Go.board_size
        for i in range(Go.board_size):
            field[i] = [Go.empty_field] * Go.board_size

        for stone in self.get_black_stones():
            x = Go.alpha_values[stone[0].upper()]
            y = int(stone[1]) - 1
            field[y][x] = Go.black_field

        for stone in self.get_white_stones():
            x = Go.alpha_values[stone[0].upper()]
            y = int(stone[1]) - 1
            field[y][x] = Go.white_field

        return field

    def get_field_as_str(self):
        field_str = ''
        field = self.get_field()
        for y in range(len(field)):
            row = field[y]
            for x in range(len(row)):
                field_str += str(row[x]) + ' '
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

    def is_finished(self):
        return self.is_finished