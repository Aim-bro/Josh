#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 2485                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/2485                           #+#        #+#      #+#     #
#    Solved: 2025/02/14 09:29:40 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #

import sys, math
input = sys.stdin.readline
N = int(input())
tree = [int(input()) for _ in range(N)]

dist = [tree[i+1]-tree[i] for i in range(N-1)]
gcd = dist[0]
for i in range(1, len(dist)):
    gcd = math.gcd(gcd, dist[i])

tot_trees = (tree[-1] - tree[0]) // gcd + 1
answer = tot_trees - N

print(answer)

    







