from epsilon import Epsilon
from params import save_interval


class Reporting:
    def __init__(self, epsilon=Epsilon(), iterations=0, report=False):
        self.iterations = iterations
        self.epsilon = epsilon
        self.round_len = 0
        self.rand = 0
        self.network = 0
        self.l_a = 0
        self.r_a = 0
        self.average_round_len = 0
        self.highest_round_len = 0
        self._report = report

    def __str__(self):
        return (
            f"Iterations: {self.iterations}\nEpsilon: {self.epsilon}\nRound length: {self.round_len}\n"
            "Random actions: {self.rand}\nNetwork actions: {self.network}\nLeft actions: {self.l_a}\n"
            "Right actions: {self.r_a}"
        )

    def update_average_round_len(self):
        if self.average_round_len == 0:
            self.average_round_len = self.round_len
        else:
            self.average_round_len = self.average_round_len + (
                (self.round_len - self.average_round_len) / (self.iterations + 1)
            )
        return self.average_round_len

    def should_save_model(self):
        return self.iterations > 10000 and self.iterations % save_interval == 0

    def report(self, train=True):
        if not self._report:
            return
        if self.iterations % 100 == 0:
            print("Highest round length:", self.highest_round_len)
            print("Average round length:", round(self.average_round_len, 2))
            print("Iterations:", self.iterations)
            if train:
                print("Epsilon:", self.epsilon.value)
                print("Left actions:", self.l_a)
                print("Right actions:", self.r_a)
                print("Random actions:", self.rand)
                print("Network actions:", self.network)
            self.reset_stats()
            print("**********************\n")

    def reset_stats(self):
        self.round_len = 0
        self.highest_round_len = 0
        self.average_round_len = 0
        self.rand = 0
        self.network = 0
        self.l_a = 0
        self.r_a = 0

    def reset_episode(self):
        if self.round_len > self.highest_round_len:
            self.highest_round_len = self.round_len
        self.round_len = 0
