#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 15552                             :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/15552                          #+#        #+#      #+#     #
#    Solved: 2024/11/12 09:37:41 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
import sys
n = int(sys.stdin.readline().rstrip())
for x in range(n):
    a, b = map(int, sys.stdin.readline().rstrip().split())
    print(a+b)