class CartPoleTask:
    def __init__(self, threshold=None):
        if threshold is None:
            self.threshold = 0.6

    def is_done(self, state):
        yaw_angle = state[2]
        return abs(yaw_angle) > self.threshold

    def get_reward(self, state):
        yaw_angle = state[2]
        if self.is_done(state):
            return -200
        return 6 - abs(yaw_angle) * 10
