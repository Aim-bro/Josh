#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 10816                             :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/10816                          #+#        #+#      #+#     #
#    Solved: 2025/02/04 15:15:26 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
import sys

N = int(sys.stdin.readline())
num_dict = {}
for num in map(int, sys.stdin.readline().split()):
    num_dict[num] = num_dict.get(num, 0) + 1

M = int(sys.stdin.readline())
print(' '.join(str(num_dict.get(x, 0)) for x in map(int, sys.stdin.readline().split())))
