__author__ = 'Marek'
import time
import random
import qbrain.util.timed_input as ti
from qbrain.go.Go import Go
from qbrain.go.QBrainGo import QBrainGo


def print_winner(go):
    if go.winner is None:
        print('draw')
    else:
        print(go.winner + ': ' + str(go.score))
    print('-' * 40)
    print()
    print()


def print_step(bw, group_name, move, field, predicted_lower_bound, predicted_upper_bound):
    player = '('
    player += group_name
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


def maybe_pause(num_seconds=5, message='Continue in 5 seconds, press enter to pause'):
    ans = ti.timed_unix_input(message, num_seconds)
    if ans is not None:
        input('press enter to continue..')


def white_stones_lost(previous_field, field):
    res = []
    for x in range(len(previous_field)):
        for y in range(len(previous_field[x])):
            if previous_field[x][y] == Go.white_field and field[x][y] != Go.white_field:
                res.append((x, y))
    return res


def black_stones_lost(previous_field, field):
    res = []
    for x in range(len(previous_field)):
        for y in range(len(previous_field[x])):
            if previous_field[x][y] == Go.black_field and field[x][y] != Go.black_field:
                res.append((x, y))
    return res


def play(brain, go, black_move_fun, white_move_fun, black_group_name, white_group_name, max_moves, board_size, maybe_pause_enabled=False):
    last_field_of_stones = go.get_field()
    stones_placed_at_move_field = [None] * board_size
    for i in range(board_size):
        stones_placed_at_move_field[i] = [0] * board_size

    move_num_black = 0
    move_num_white = 0

    while not go.is_finished:
        bw = go.next
        field = go.get_field()

        if bw == Go.black_str:
            move, predicted_lower_bound, predicted_upper_bound = black_move_fun(go, black_group_name, field, move_num_black, True)

            if move[0] is not None:
                stones_placed_at_move_field[move[0][0]][move[0][1]] = move_num_black
            else:
                brain.post_reward(black_group_name, -3, move_num_black, 1)

            move_num_black += 1
        else:
            move, predicted_lower_bound, predicted_upper_bound = white_move_fun(go, white_group_name, field, move_num_white, False)

            if move[0] is not None:
                stones_placed_at_move_field[move[0][0]][move[0][1]] = move_num_white
            else:
                brain.post_reward(white_group_name, -3, move_num_white, 1)
            move_num_white += 1

        # now_field = go.get_field()
        # white_lost = white_stones_lost(last_field_of_stones, now_field)
        # black_lost = black_stones_lost(last_field_of_stones, now_field)
        # last_field_of_stones = now_field

        # for i in range(len(white_lost)):
        #     placed_stone_at = stones_placed_at_move_field[white_lost[i][0]][white_lost[i][1]]
        #     brain.post_reward(white_group_name, -10.0, placed_stone_at, move_num_white - placed_stone_at)
        #     brain.post_reward(black_group_name, 1.0, placed_stone_at, move_num_black - placed_stone_at)
        #
        # for i in range(len(black_lost)):
        #     placed_stone_at = stones_placed_at_move_field[black_lost[i][0]][black_lost[i][1]]
        #     brain.post_reward(black_group_name, -10.0, placed_stone_at, move_num_black - placed_stone_at)
        #     brain.post_reward(white_group_name, 1.0, placed_stone_at, move_num_white - placed_stone_at)

        field_str = go.get_field_as_str()
        if Go.black_str == bw:
            print_step(bw, black_group_name, move, field_str, predicted_lower_bound, predicted_upper_bound)
        else:
            print_step(bw, white_group_name, move, field_str, predicted_lower_bound, predicted_upper_bound)

        if move_num_black + move_num_white >= max_moves:
            go.finish_game()

        if maybe_pause_enabled:
            maybe_pause(num_seconds=10, message='Next move in 10 seconds, press enter for pause..')
    print_winner(go)

    if go.winner == Go.black_str:
        brain.post_reward(black_group_name, go.score, 0, move_num_black)
        brain.post_reward(white_group_name, -go.score, 0, move_num_white)
    elif go.winner == Go.white_str:
        brain.post_reward(black_group_name, -go.score, 0, move_num_black)
        brain.post_reward(white_group_name, go.score, 0, move_num_white)

    brain.flush_group(black_group_name)
    brain.flush_group(white_group_name)


def move_ind_to_move(move_ind, board_size, pass_move_ind):
    if move_ind == pass_move_ind:
        move = (None, Go.pass_str)
    else:
        x = int(move_ind / board_size)
        y = int(move_ind % board_size)
        move = ((x, y), None)
    return move


def move_to_move_ind(move, board_size, pass_move_ind):
    if move[0] is None:
        move_ind = pass_move_ind
    else:
        move_ind = move[0][0] * board_size + move[0][1]
    return int(move_ind)


def get_vs_str(is_black_gnugo, is_white_gnugo):
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

    return vs_string


class GoApp():
    replay_experience_prefix = '__replay__'

    def __init__(self, board_size=9):
        self.board_size = board_size
        self.pass_move_ind = board_size * board_size
        self.brain = QBrainGo(board_size, [(3, 8)], [2048, 1024, 1024],
                              'saves_mem/', 'go_autosave', '.pkl',
                              'saves_net/', 'go_autosave', '.ckpt')
        self.brain.load()
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
                                                                                        field,
                                                                                        possible_moves,
                                                                                        move_num,
                                                                                        is_black)
        net_move = move_ind_to_move(net_move_ind, self.board_size, self.pass_move_ind)

        if net_move[0] is None:
            go.move_pass()
        else:
            go.move(net_move[0][0], net_move[0][1])

        return net_move, predicted_lower_bound, predicted_upper_bound

    def play_expert_move(self, go, group_name, field, move_num, is_black):
        expert_move = go.expert_move()

        move_ind = move_to_move_ind(expert_move, self.board_size, self.pass_move_ind)

        self.brain.expert_forward(group_name, field, move_ind, move_num, is_black)
        return expert_move, 0, 0

    def play(self, is_black_gnugo=False, is_white_gnugo=False, max_moves=800, auto_replay=True, maybe_pause_enabled=False):
        go = Go(self.board_size)

        vs_string = get_vs_str(is_black_gnugo, is_white_gnugo)

        black_group_name = Go.black_str + '_' + vs_string + '_' + str(self.mem_index)
        white_group_name = Go.white_str + '_' + vs_string + '_' + str(self.mem_index)
        self.mem_index += 1

        if is_black_gnugo:
            black_move_fun = self.play_expert_move
        else:
            black_move_fun = self.play_net_move

        if is_white_gnugo:
            white_move_fun = self.play_expert_move
        else:
            white_move_fun = self.play_net_move

        play(brain=self.brain, go=go,
             black_move_fun=black_move_fun, white_move_fun=white_move_fun,
             black_group_name=black_group_name, white_group_name=white_group_name,
             board_size=self.board_size, max_moves=max_moves, maybe_pause_enabled=maybe_pause_enabled)

        go.close()

        if auto_replay:
            self.replay_with_random_move(black_group_name, white_group_name, num_moves_backward=10)

    def replay_all_experiences(self, is_black_gnugo=True, is_white_gnugo=True, max_moves=8000, num_moves_backward=4, num_replays_per_experience=5):
        for experience_group_name in list(self.brain.mem.flushed_experience_groups):
            if not experience_group_name.startswith(GoApp.replay_experience_prefix) and experience_group_name.find(Go.white_str) < 0:
                black_group_name = experience_group_name
                white_group_name = black_group_name.replace(Go.black_str, Go.white_str)

                self.replay_with_random_move(black_group_name=black_group_name, white_group_name=white_group_name,
                                             is_black_gnugo=is_black_gnugo, is_white_gnugo=is_white_gnugo,
                                             max_moves=max_moves, num_moves_backward=num_moves_backward,
                                             num_replays=num_replays_per_experience)

    def replay_with_random_move(self, black_group_name, white_group_name, is_black_gnugo=True, is_white_gnugo=True, max_moves=8000, num_moves_backward=4, num_replays=5):

        black_group = self.brain.mem.load_group(black_group_name)
        white_group = self.brain.mem.load_group(white_group_name)

        vs_string = get_vs_str(is_black_gnugo, is_white_gnugo)

        for replay_num in range(num_replays):
            group_name_prefix = GoApp.replay_experience_prefix + str(replay_num) + '_' + vs_string + '_back_' + str(num_moves_backward) + '_'

            black_group_name_replay = group_name_prefix + black_group_name
            white_group_name_replay = group_name_prefix + white_group_name

            if is_black_gnugo:
                move_fun = self.play_expert_move
            else:
                move_fun = self.play_net_move

            last_replay_move = black_group.last - num_moves_backward
            black_replayer = Replayer(brain=self.brain,
                                      experience_group=black_group, last_replay_move_num=last_replay_move,
                                      move_fun=move_fun,
                                      board_size=self.board_size, pass_move_ind=self.pass_move_ind)

            if is_white_gnugo:
                move_fun = self.play_expert_move
            else:
                move_fun = self.play_net_move

            last_replay_move = white_group.last - num_moves_backward
            white_replayer = Replayer(brain=self.brain,
                                      experience_group=white_group, last_replay_move_num=last_replay_move,
                                      move_fun=move_fun,
                                      board_size=self.board_size, pass_move_ind=self.pass_move_ind)
            go = Go(self.board_size)
            play(brain=self.brain, go=go, max_moves=max_moves, board_size=self.board_size,
                 black_move_fun=black_replayer.play_move, black_group_name=black_group_name_replay,
                 white_move_fun=white_replayer.play_move, white_group_name=white_group_name_replay)
            
    def net_only(self, maybe_pause_enabled=False):
        self.play(False, False, maybe_pause_enabled=maybe_pause_enabled)

    def expert_only(self, maybe_pause_enabled=False):
        self.play(True, True, maybe_pause_enabled=maybe_pause_enabled)

    def save(self):
        self.brain.save()

    def load(self):
        self.brain.load()

    def train(self, batch_size=1024, num_iter=10, max_err=0.0):
        self.brain.train(batch_size, num_iter, max_err, None)

    def play_and_train(self, num_cycle=10, batch_size=1024, num_iter=2, num_batches=2, max_moves=800, maybe_pause_enabled=False):
        for i in range(num_cycle):
            self.play(is_black_gnugo=True, is_white_gnugo=False, max_moves=max_moves, maybe_pause_enabled=maybe_pause_enabled)
            maybe_pause()
            self.play(is_black_gnugo=True, is_white_gnugo=True, max_moves=max_moves, maybe_pause_enabled=maybe_pause_enabled)
            maybe_pause()
            self.play(is_black_gnugo=False, is_white_gnugo=True, max_moves=max_moves, maybe_pause_enabled=maybe_pause_enabled)
            maybe_pause()
            self.play(is_black_gnugo=False, is_white_gnugo=False, max_moves=max_moves, maybe_pause_enabled=maybe_pause_enabled)
            maybe_pause()
            for batch_num in range(num_batches):
                self.train(batch_size=batch_size, num_iter=num_iter)
        self.save()

    def play_net_and_train(self, num_cycle=100, batch_size=1024, num_iter=4, num_batches=8, max_moves=800, maybe_pause_enabled=False):
        for i in range(num_cycle):
            self.play(is_black_gnugo=False, is_white_gnugo=False, max_moves=max_moves, maybe_pause_enabled=maybe_pause_enabled)
            maybe_pause()
            for batch_num in range(num_batches):
                self.train(batch_size=batch_size, num_iter=num_iter)
        self.save()


class Replayer():
    def __init__(self, brain, experience_group, last_replay_move_num, move_fun, board_size, pass_move_ind):
        self.brain = brain
        self.experience_group = experience_group
        self.last_replay_move_num = last_replay_move_num
        self.move_fun = move_fun
        self.board_size = board_size
        self.pass_move_ind = pass_move_ind

    def play_move(self, go, group_name, field, move_num, is_black):

        lower = upper = 0
        if move_num <= self.last_replay_move_num and move_num in self.experience_group.group:
            exp = self.experience_group.group[move_num]
            move = move_ind_to_move(exp.action, self.board_size, self.pass_move_ind)
            if move[0] is None:
                go.move_pass()
            else:
                go.move(move[0][0], move[0][1])

            self.brain.expert_forward(group_name, field, exp.action, move_num, is_black)

        elif move_num == self.last_replay_move_num + 1:
            if is_black:
                possible_moves = go.get_black_possible_moves_list()
            else:
                possible_moves = go.get_white_possible_moves_list()

            rnd_index = random.randint(0, len(possible_moves) + 1)
            if rnd_index < len(possible_moves):
                move = (possible_moves[rnd_index], None)
            else:
                move = (None, Go.pass_str)

            print('random move: ')
            print(possible_moves)
            print(rnd_index)
            print(move)

            if move[0] is None:
                go.move_pass()
            else:
                go.move(move[0][0], move[0][1])

            self.brain.expert_forward(group_name, field, move_to_move_ind(move, self.board_size, self.pass_move_ind), move_num, is_black)

        else:
            move, lower, upper = self.move_fun(go, group_name, field, move_num, is_black)

        return move, lower, upper