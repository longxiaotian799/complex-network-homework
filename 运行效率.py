# # lis = [1, 3, 4 ,5 ,6, 4,6 ,4 ]
# # for i,v in enumerate(lis)


# # print([i for i in range(1,101) if (i % 7 == 0 and i % 5 != 0)])
# # import numpy as np
# # m = np.zeros((9,9))
# # for i in range(1,10):
# #     for j in range(i, 10):
# #         print(f'{j}*{i}={i*j}', sep=' ')
# #         m[j, i] = f'{j}*{i}={i*j}'
# # print(m)

# # for i in range(1, 10):
# #     for j in range(1,i+1):
# #         print(f"{i}*{j}={i*j}", end=" ")
# #         print()
# # import time

# # start_time = time.time()

# # sum1 = 0
# # for i in range(1, 100000):
# #         sum1 += i
# # print(sum1)

# # end_time = time.time()

# # print('程序运行时间为：', end_time - start_time, '秒')



# # import time

# # start_time = time.time()

# # print(sum(list(range(1, 100000))))

# # end_time = time.time()

# # print('程序运行时间为：', end_time - start_time, '秒')

# for i in range(1, 3):
#     if i == 2:
#         break
# else:
#     print('ok')

import datetime

date = '2023-04-12'
now = datetime.datetime.strptime(date, '%Y-%m-%d')
start = datetime.datetime.strptime(date[:4]+'-01-01', '%Y-%m-%d')
print('今天是今年的第', (now - start).days + 1, '天')

