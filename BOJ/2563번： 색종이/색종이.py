#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 2563                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/2563                           #+#        #+#      #+#     #
#    Solved: 2024/11/20 15:13:25 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #

white = [[0] * 100 for _ in range(100)]

n = int(input())

for _ in range(n):
    x, y = map(int, input().split())
    for i in range(x, x + 10):
        for j in range(y, y + 10):
            white[i][j] = 1

area = sum(row.count(1) for row in white)

print(area)
