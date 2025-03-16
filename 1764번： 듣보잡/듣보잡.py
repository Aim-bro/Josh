#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 1764                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/1764                           #+#        #+#      #+#     #
#    Solved: 2025/02/04 16:48:08 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
import sys
N, M = map(int, sys.stdin.readline().split())
name_dict = {}
for _ in range(N):
    name_dict[sys.stdin.readline().strip()] = 1

for _ in range(M):
    name = sys.stdin.readline().strip()
    if name in name_dict:
        name_dict[name] += 1
    else:
        name_dict[name] = 1

result = [x for x, y in name_dict.items() if y == 2]
print(len(result))
print('\n'.join(sorted(result)))
