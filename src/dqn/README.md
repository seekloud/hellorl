# 工作文档

### 目标
1. 完成 q-learning 的基本功能;
2. 在 atari 游戏：'Riverraid' 或 'MsPacman' 中进行实验;
3. 进行一定的调优;


### 步骤
1. 阅读rl相关blog 与 李宏毅rl视频;
2. 阅读github上相关代码 [mxnet实现的q-learning](https://github.com/zmonoid/mxdqn)
3. 设计软件架构;
4. 实现代码, 跑通程序, 完成训练与测试;
5. 调优;


### 主要模块
1. replay store
2. q-network
3. 游戏玩家(player)封装
3. 游戏封装环境(environment)封装
3. 训练(测试)流程封装
4. 结果绘制







### 2018-08-31 16:49
1. 每局开启时增加随机个0操作;
2. 增加梯度裁剪; 
3. update_target_net变间距更新






















