class C:
    def __init__(self):
        self.a = 4
        self.b = 3

    def a(self, val):
        return val + 3


c = C()
print(c.a)
print(c.a(10))
