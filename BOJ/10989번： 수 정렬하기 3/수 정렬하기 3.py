#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 10989                             :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/10989                          #+#        #+#      #+#     #
#    Solved: 2025/01/08 14:57:39 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
import sys
n = int(sys.stdin.readline())
# nums = [int(sys.stdin.readline()) for _ in range(n)]
# 입력 크기는 매우 크고 메모리 제한이 있기 때문에 메모리 제한을 고려해야 함
cnt = [0] * 10001
for _ in range(n):
    cnt[int(sys.stdin.readline())] += 1

for i in range(10001):
    if cnt[i] != 0:
        for _ in range(cnt[i]):
            print(i)

