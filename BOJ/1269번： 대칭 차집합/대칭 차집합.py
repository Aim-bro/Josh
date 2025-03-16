#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 1269                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/1269                           #+#        #+#      #+#     #
#    Solved: 2025/02/05 09:32:43 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
import sys
input = sys.stdin.readline
N, M = map(int, input().split())

set_A = set(map(int, input().split()))
set_B = set(map(int, input().split()))

print(len(set_A - set_B) + len(set_B - set_A))

