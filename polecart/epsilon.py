class Epsilon:
    def __init__(self, start=1.0, end=0.01, decay=0.9995):
        self.start = start
        self.end = end
        self.decay = decay
        self._value = start

    def decay_epsilon(self):
        self._value *= self.decay
        if self._value < self.end:
            self._value = self.end
        return self._value

    @property
    def value(self):
        return self._value

    def get_epsilon(self):
        return self._value

    def set_epsilon(self, value):
        self._value = value

    def reset_epsilon(self):
        self._value = self.start

    def __str__(self):
        return f"Epsilon: {self._value}"
