#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 9063                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/9063                           #+#        #+#      #+#     #
#    Solved: 2024/12/09 15:16:52 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
N = int(input())
x_p = []
y_p = []
for _ in range(N):
    x, y = map(int,input().split())
    x_p.append(x)
    y_p.append(y)
if N > 1:
    x_l = max(x_p) - min(x_p)
    y_l = max(y_p) - min(y_p)
    print(x_l*y_l)
else:
    print(0)
