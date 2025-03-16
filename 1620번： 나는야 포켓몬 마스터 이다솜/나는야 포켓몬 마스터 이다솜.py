#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 1620                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/1620                           #+#        #+#      #+#     #
#    Solved: 2025/02/03 15:40:46 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
import sys
N, M = map(int, sys.stdin.readline().split())

dic_name = {}
dic_num = {}

for i in range(N):
    name = sys.stdin.readline().strip()
    dic_name[name] = i+1
    dic_num[i+1] = name

for _ in range(M):
    Q = sys.stdin.readline().strip()
    if Q.isdigit():
        print(dic_num[int(Q)])
    else:
        print(dic_name[Q])
        