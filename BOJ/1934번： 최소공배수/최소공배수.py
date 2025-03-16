#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 1934                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/1934                           #+#        #+#      #+#     #
#    Solved: 2025/02/10 13:08:21 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
import sys
import math
input = sys.stdin.readline


for _ in range(int(input())):
    A, B = map(int, input().split())
    print(A*B//math.gcd(A, B))
