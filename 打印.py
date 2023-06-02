# 打印由星号组成的菱形，而且可以灵活控制图案的大小
n = int(input('输入一个n的值：'))
for i in range(1, n, 1):
    # center() 返回一个原字符串居中,并使用空格填充至长度 width 的新字符串。默认填充字符为空格
    print((' * ' * i).center(n * 3))#打印上半部分
for i in range(n, 0, -1):#打印最长的一行以及 下半部分
    print((' * ' * i).center(n * 3))