

class action_done:
    def __init__(self):
        print('initiate action_done')
        self.action = []
        self.random_state = []

        self.action_rdmstate = None

    def append(self, action, random_state):
        """
        action: str
        random_state: class: tuple
        """
        self.action.append(action)
        self.random_state.append(random_state)

        self.action_rdmstate = {'action_done': self.action,
                                'random_state': self.random_state}
        return

