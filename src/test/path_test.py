# Author: Taoz
# Date  : 9/13/2018
# Time  : 11:17 PM
# FileName: path_test.py


import os

p = os.path.abspath(__file__)
p1 = os.path.abspath('conf_1.ini')
p2 = os.path.abspath('conf_999.ini')

print(__file__)
print(p)
print(p1)
print(p2)
print(type(p))
