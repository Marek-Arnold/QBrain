__author__ = 'Marek'
import gtp


class Go():
    alpha_positions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
    black_str = 'black'
    white_str = 'white'
    pass_str = 'pass'

    def __init__(self):
        self.go = gtp.GoTextPipe()
        self.next = Go.black_str
        self.last_has_passed = False
        self.is_finished = False

    def switch_next(self):
        if self.next == Go.black:
            self.next = Go.white
        else:
            self.next = Go.black

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