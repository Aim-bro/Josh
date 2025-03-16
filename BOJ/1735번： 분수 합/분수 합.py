#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 1735                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/1735                           #+#        #+#      #+#     #
#    Solved: 2025/02/12 12:26:12 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
import sys
import math
input = sys.stdin.readline

A, B = map(int, input().split())
C, D = map(int, input().split())

y = B*D//math.gcd(B,D)
x = A*(y//B) + C*(y//D)

print(x//math.gcd(x,y), y//math.gcd(x,y))



