__author__ = 'Marek'
from Go import Go


class GoApp():
    def __init_(self):

    def expert_only(self):
        go = Go()
        while(not go.is_finished):
            go.expert_move()
            print(go.show_board())