# Author: Taoz
# Date  : 9/12/2018
# Time  : 9:43 PM
# FileName: conf_test.py



import configparser
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.sections()


config.read('conf_default.ini')
config.read('conf_1.ini')
sections = config.sections()
print(sections)


d1 = config['D1']

print(d1.get('lra'))
print(d1.get('wda'))
print(d1.get('lr'))
print(d1.get('wd'))

q = config['q_learning']

print(q.get('lra'))
print(q.get('wda'))
print(q.get('lr'))
print(q.get('wd'))

