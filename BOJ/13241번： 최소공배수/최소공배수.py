#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 13241                             :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/13241                          #+#        #+#      #+#     #
#    Solved: 2025/02/11 22:30:17 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
import sys
import math
input = sys.stdin.readline

A, B = map(int, input().split())
print(A*B//math.gcd(A, B))
print(math.gcd(A, B))
