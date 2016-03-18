import falcon
import json
import os
from QBrain import QBrain

num_input_wall_distance = 2 + 1 * 4
num_sensor_enemy = 16
num_input_enemy = num_sensor_enemy * 4
num_input_hit_by_bullet_damage = 3
num_inputs = num_input_wall_distance + num_input_enemy + num_input_hit_by_bullet_damage
num_actions = 6
temporal_window = 60 
single_input_size = num_inputs + num_actions
num_neurons_in_convolution_layers = [128, 64, 32]
num_neurons_in_fully_connected_layers = [1024, 512, 256]

brain = QBrain(single_input_size, temporal_window, num_actions, num_neurons_in_convolution_layers,
               num_neurons_in_fully_connected_layers)

if os.path.exists('saves/'):
    files = os.listdir('saves/')
    brain.load('xx_autosave')


class ForwardResource:
    def on_post(self, req, resp):
        print('forward')
        body = req.stream.read()
        body = json.loads(body)
        array_with_num_inputs_numbers = []
        for i in range(0, num_input_wall_distance):
            ind = 'dw_' + str(i)
            if ind in body:
                d = float(body[ind])
            else:
                d = 0
            array_with_num_inputs_numbers.append(d)

        for i in range(0, num_sensor_enemy):
            ind = 'de_' + str(i)
            if ind in body:
                d = float(body[ind])
		e = float(body['ee_' + str(i)])
		h = float(body['he_' + str(i)])
		v = float(body['ve_' + str(i)])
            else:
                d = 0
		e = 0
		h = 0
		v = 0
            array_with_num_inputs_numbers.append(d)
            array_with_num_inputs_numbers.append(e)
            array_with_num_inputs_numbers.append(h)
            array_with_num_inputs_numbers.append(v)

        for i in range(0, num_input_hit_by_bullet_damage):
            ind = 'da_' + str(i)
            if ind in body:
                d = float(body[ind])
            else:
                d = 0
            array_with_num_inputs_numbers.append(d)

	print(str(len(array_with_num_inputs_numbers)) + ' of ' + str(num_inputs))

        group_name = 'default'
        if 'gn' in body:
            group_name = body['gn']

        time = int(body['t'])

        action = brain.forward(group_name, array_with_num_inputs_numbers, time)
        data = {'action': action}

        resp.body = json.dumps(data)


class ExpertForwardResource:
    def on_post(self, req, resp):
        print('expert_forward')
        body = req.stream.read()
        body = json.loads(body)
        array_with_num_inputs_numbers = []
        for i in range(0, num_input_wall_distance):
            ind = 'dw_' + str(i)
            if ind in body:
                d = float(body[ind])
            else:
                d = 0
            array_with_num_inputs_numbers.append(d)

        for i in range(0, num_sensor_enemy):
            ind = 'de_' + str(i)
            if ind in body:
                d = float(body[ind])
		e = float(body['ee_' + str(i)])
		h = float(body['he_' + str(i)])
		v = float(body['ve_' + str(i)])
            else:
                d = 0
		e = 0
		h = 0
		v = 0
            array_with_num_inputs_numbers.append(d)
            array_with_num_inputs_numbers.append(e)
            array_with_num_inputs_numbers.append(h)
            array_with_num_inputs_numbers.append(v)


        for i in range(0, num_input_hit_by_bullet_damage):
            ind = 'da_' + str(i)
            if ind in body:
                d = float(body[ind])
            else:
                d = 0
            array_with_num_inputs_numbers.append(d)

	print(str(len(array_with_num_inputs_numbers)) + ' of ' + str(num_inputs))

        group_name = 'default'
        if 'gn' in body:
            group_name = body['gn']

        time = int(body['t'])

        str_action = body['a']
        action = 5
        if str_action == 'FORWARD_LEFT':
            action = 0
        elif str_action == 'FORWARD':
            action = 1
        elif str_action == 'FORWARD_RIGHT':
            action = 2
        elif str_action == 'BACKWARD_LEFT':
            action = 3
        elif str_action == 'BACKWARD':
            action = 4

        brain.expert_forward(group_name, array_with_num_inputs_numbers, action, time)


class TrainResource:
    def on_post(self, req, resp):
        print('train')
        body = req.stream.read()
        body = json.loads(body)
        num_iter = 32
        if 'ni' in body:
            num_iter = int(body['ni'])

        batch_size = 32
        if 'bs' in body:
            batch_size = int(body['bs'])

        brain.train(batch_size, num_iter)


class RewardResource:
    def on_post(self, req, resp):
        print('reward')
        body = req.stream.read()
        body = json.loads(body)
        group_name = 'default'
        if 'gn' in body:
            group_name = body['gn']

        reward = 0
        if 'r' in body:
            reward = float(body['r'])

        start = 0
        if 's' in body:
            start = int(body['s'])

        duration = 1
        if 'd' in body:
            duration = max(int(body['d']), 1)

        brain.post_reward(group_name, reward, start, duration)


class FlushGroupResource:
    def on_post(self, req, resp):
        print('flush')
        body = req.stream.read()
        body = json.loads(body)
        group_name = 'default'
        if 'gn' in body:
            group_name = body['gn']
        brain.flush_group(group_name)


class SaveResource:
    def on_post(self, req, resp):
        print('save')
        body = req.stream.read()
        body = json.loads(body)
        model_name = 'default'
        if 'mn' in body:
            model_name = body['mn']
        brain.save(model_name)


class LoadResource:
    def on_post(self, req, resp):
        print('load')
        body = req.stream.read()
        body = json.loads(body)
        model_name = 'default'
        if 'mn' in body:
            model_name = body['mn']
        brain.load(model_name)


class TestResource:
    def on_get(self, req, resp):
        print('test')
        resp.body = 'hello world'


api = falcon.API()
api.add_route('/forward', ForwardResource())
api.add_route('/expert_forward', ExpertForwardResource())
api.add_route('/train', TrainResource())
api.add_route('/reward', RewardResource())
api.add_route('/flush_group', FlushGroupResource())
api.add_route('/save', SaveResource())
api.add_route('/load', LoadResource())
api.add_route('/test', TestResource())
