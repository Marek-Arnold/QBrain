__author__ = 'Marek'
import time
from qbrain.go.Go import Go
from qbrain.go.QBrainGo import QBrainGo


def print_winner(go):
    print(go.winner + ': ' + str(go.score))
    print('-' * 40)
    print()
    print()


def print_step(bw, is_gnugo, move, field, predicted_lower_bound, predicted_upper_bound):
    player = '('
    if is_gnugo:
        player += 'gnugo'
    else:
        player += 'net'
    player += ')'
    print(bw + ' ' + player + ': ' + str(move) + '\tlower: ' + str(predicted_lower_bound) + '\tupper: ' + str(predicted_upper_bound))
    print()
    print(field)
    print('-' * 20)
    print()


def flatten_field(field):
    flat_field = []

    for i in range(len(field)):
        flat_field.extend(field[i])
    return flat_field


class GoApp():
    def __init__(self, board_size=19):
        self.board_size = board_size
        self.pass_move_ind = board_size * board_size
        self.brain = QBrainGo(board_size, [(3, 8), (5, 12), (8, 16)], [4096, 2048])
        self.brain.load('go_autosave')
        millis = int(round(time.time() * 1000))
        self.mem_index = millis
        print('Ready..')

    def play_net_move(self, go, group_name, field, move_num, is_black):
        possible_moves = []

        if is_black:
            pm = go.get_black_possible_moves()
        else:
            pm = go.get_white_possible_moves()

        possible_moves.extend(flatten_field(pm))
        possible_moves.append(1.0)
        net_move_ind, predicted_lower_bound, predicted_upper_bound = self.brain.forward(group_name,
                                                                                        flatten_field(field),
                                                                                        possible_moves,
                                                                                        move_num,
                                                                                        is_black)
        if net_move_ind == self.pass_move_ind:
            net_move = (None, Go.pass_str)
            go.move_pass()
        else:
            x = net_move_ind % self.board_size
            y = int(net_move_ind / self.board_size)
            net_move = ((x, y), None)
            go.move(x, y)

        return net_move, predicted_lower_bound, predicted_upper_bound

    def play_expert_move(self, go, group_name, field, move_num, is_black):
        expert_move = go.expert_move()

        if expert_move[0] is None:
            move_ind = self.pass_move_ind
        else:
            move_ind = expert_move[0][0] + expert_move[0][1] * self.board_size

        self.brain.expert_forward(group_name, flatten_field(field), move_ind, move_num, is_black)
        return expert_move, 0, 0

    def play(self, is_black_gnugo=False, is_white_gnugo=False):
        go = Go()
        move_num_black = 0
        move_num_white = 0

        vs_string = ''
        if is_black_gnugo:
            vs_string += 'gnugo'
        else:
            vs_string += 'net'
        vs_string += '_vs_'

        if is_white_gnugo:
            vs_string += 'gnugo'
        else:
            vs_string += 'net'

        black_group_name = Go.black_str + '_' + vs_string + '_' + str(self.mem_index)
        white_group_name = Go.white_str + '_' + vs_string + '_' + str(self.mem_index)
        self.mem_index += 1

        while not go.is_finished:
            bw = go.next
            field = go.get_field()

            if bw == Go.black_str:
                if is_black_gnugo:
                    is_gnugo = True
                    move, predicted_lower_bound, predicted_upper_bound = self.play_expert_move(go, black_group_name, field, move_num_black, True)
                else:
                    is_gnugo = False
                    move, predicted_lower_bound, predicted_upper_bound = self.play_net_move(go, black_group_name, field, move_num_black, True)
                move_num_black += 1
            else:
                if is_white_gnugo:
                    is_gnugo = True
                    move, predicted_lower_bound, predicted_upper_bound = self.play_expert_move(go, black_group_name, field, move_num_white, False)
                else:
                    is_gnugo = False
                    move, predicted_lower_bound, predicted_upper_bound = self.play_net_move(go, white_group_name, field, move_num_white, False)
                move_num_white += 1

            field_str = go.get_field_as_str()
            print_step(bw, is_gnugo, move, field_str, predicted_lower_bound, predicted_upper_bound)
        print_winner(go)

        if go.winner == Go.black_str:
            self.brain.post_reward(black_group_name, go.score, 0, move_num_black)
            self.brain.post_reward(white_group_name, -go.score, 0, move_num_white)
        else:
            self.brain.post_reward(black_group_name, -go.score, 0, move_num_black)
            self.brain.post_reward(white_group_name, go.score, 0, move_num_white)

        self.brain.flush_group(black_group_name)
        self.brain.flush_group(white_group_name)

    def net_only(self):
        self.play(False, False)

    def expert_only(self):
        self.play(True, True)

    def save(self, name='go_autosave'):
        self.brain.save(name)

    def load(self, name='go_autosave'):
        self.brain.load(name)

    def train(self, batch_size=1024, num_iter=10, max_err=0.0):
        self.brain.train(batch_size, num_iter, max_err, None)

    def play_and_train(self, num_cycle=10, batch_size=4096, num_iter=2, num_batches=2):
        for i in range(num_cycle):
            self.play(is_black_gnugo=True, is_white_gnugo=False)
            self.play(is_black_gnugo=True, is_white_gnugo=True)
            self.play(is_black_gnugo=False, is_white_gnugo=True)
            self.play(is_black_gnugo=False, is_white_gnugo=False)
            for batch_num in range(num_batches):
                self.train(batch_size=batch_size, num_iter=num_iter)
        self.save()