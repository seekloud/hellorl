# Author: Taoz
# Date  : 9/12/2018
# Time  : 9:43 PM
# FileName: conf_test.py


import configparser

config = configparser.ConfigParser(allow_no_value=True, interpolation=configparser.ExtendedInterpolation())
config.sections()

config.read('D:\workstation\python\hellorl\src\\test\conf_default.ini')
# config.read('conf_default.ini')
config.read('conf_1.ini')
sections = config.sections()
print(sections)

d1 = config['D1']

print(d1.get('lra'))
print(d1.get('wda'))
print(d1.get('lr'))
print(d1.get('wd'))
print('c=', d1.get('c'))
print('c type' + str(type(d1.get('c'))))
print(d1.get('d'))

print('-----------------------------------------------------')

q = config['q_learning']

print(q.get('lra'))
print(q.get('wda'))
print(q.get('lr'))
print(q.get('wd'))
print('items:')
for k, v in q.items():
    print('%s = %s' % (k, v))

dqn_configuration = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
dqn_configuration.sections()
dqn_configuration.read('../../configurations/dqn_default_conf.ini')
dqn_configuration.read('../../configurations/dqn_conf_1001.ini')

dqn_conf = dqn_configuration['DQN']

print('-----------------------------------------------------')
print(dqn_conf.get('a'))
print(dqn_conf.get('b'))
print(dqn_conf.get('c'))
print(dqn_conf.get('d'))
