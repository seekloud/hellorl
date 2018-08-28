# Author: Taoz
# Date  : 8/27/2018
# Time  : 11:33 PM
# FileName: test1.py




class A(object):
    def __init__(self, v):
        self.v = v
        pass

    def _add_one(self):
        self.v += 1

    def _add(self, v):
        self.v += v

    def add_other(self, other):
        other._add(self.v)



if __name__ == '__main__':
    a = A(10)
    b = A(5)
    b.add_other(a)
    b._add(100)
    print(a.v)
    print(b.v)

