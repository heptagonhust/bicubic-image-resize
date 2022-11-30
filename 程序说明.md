# 程序说明

### 环境配置：

设备环境

![image-20221201023800496](C:\Users\hp\AppData\Roaming\Typora\typora-user-images\image-20221201023800496.png)

IDE：Clion 2021.3（自带cmake配置工具）

### 目前效果：

![image-20221201023957044](C:\Users\hp\AppData\Roaming\Typora\typora-user-images\image-20221201023957044.png)

### 优化策略：

1、通过Openmp进行并行加速，效果提升显著

2、通过sse指令集进行优化，效果提升较为显著

3、下一步考虑使用GPU进行加速（未实现）

### 收获：

学习到了很多高性能计算的策略，包括一些指令集和超算的一些基本概念，虽然没有能够全部实现，但是希望能进一步去学习，去把每一个策略实现一下，希望有机会能进入超算队和大佬们一起学习！！！