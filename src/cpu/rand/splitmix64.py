class Splitmix64:
    def __init__(self, seed=0):
        self.state = seed

    def next(self):
        x = self.state
        self.state += 0x9e3779b97f4a7c15
        z = x
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb
        return z ^ (z >> 31)

    def rand(self):
        return self.next() / 0xffffffffffffffff

prng = Splitmix64(0x874384E28A4BC0D6)
print(hex(prng.rand()))

# First #: 3AC54D35EB8CCCE250E87ABFBD92334E