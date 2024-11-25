#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 2566                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/2566                           #+#        #+#      #+#     #
#    Solved: 2024/11/20 09:48:18 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #

max_val = -1
max_row = 0
max_col = 0

for x in range(9):
    result = [int(x) for x in input().split()]
    current_max = max(result)
    if current_max > max_val:
        max_val = current_max
        max_row = x + 1
        max_col = result.index(current_max) + 1

print(max_val)
print(max_row, max_col)
