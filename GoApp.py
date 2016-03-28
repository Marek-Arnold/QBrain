__author__ = 'Marek'
from qbrain.go.Go import Go
from qbrain.go.QBrainGo import QBrainGo


def print_winner(go):
    print(go.winner + ': ' + str(go.score))
    print('-' * 40)
    print()
    print()


def print_step(bw, move, field):
    print(bw + str(move))
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
        self.brain = QBrainGo(board_size, [(3, 4), (5, 8), (8, 8)], [4096, 2048])
        self.brain.load('go_autosave')
        self.mem_index = 0
        print('Ready..')

    def net_only(self):
        go = Go()
        move_num_black = 0
        move_num_white = 0

        black_group_name = Go.black_str + '_' + str(self.mem_index)
        white_group_name = Go.white_str + '_' + str(self.mem_index)
        self.mem_index += 1

        while not go.is_finished:
            bw = go.next
            field = go.get_field()

            if bw == Go.black_str:
                possible_moves = []
                pm = go.get_black_possible_moves()
                possible_moves.extend(flatten_field(pm))
                possible_moves.append(1.0)
                net_move_ind = self.brain.forward(black_group_name, flatten_field(field), possible_moves,
                                                  move_num_black, True)
                move_num_black += 1
            else:
                possible_moves = []
                possible_moves.extend(flatten_field(go.get_white_possible_moves()))
                possible_moves.append(1.0)
                net_move_ind = self.brain.forward(white_group_name, flatten_field(field), possible_moves,
                                                  move_num_white, False)
                move_num_white += 1

            if net_move_ind == self.pass_move_ind:
                net_move = (None, Go.pass_str)
                go.move_pass()
            else:
                x = net_move_ind % self.board_size
                y = int(net_move_ind / self.board_size)
                net_move = ((x, y), None)
                print(net_move)
                go.move(x, y)

            field_str = go.get_field_as_str()
            print_step(bw, net_move, field_str)
        print_winner(go)

        if go.winner == Go.black_str:
            self.brain.post_reward(black_group_name, go.score, 0, move_num_black)
            self.brain.post_reward(white_group_name, -go.score, 0, move_num_white)
        else:
            self.brain.post_reward(black_group_name, -go.score, 0, move_num_black)
            self.brain.post_reward(white_group_name, go.score, 0, move_num_white)

        self.brain.flush_group(black_group_name)
        self.brain.flush_group(white_group_name)
        self.brain.save('go_autosave')

    def expert_only(self):
        go = Go()
        while not go.is_finished:
            bw = go.next
            expert_move = go.expert_move()
            field = go.get_field_as_str()
            print_step(bw, expert_move, field)
        print_winner(go)