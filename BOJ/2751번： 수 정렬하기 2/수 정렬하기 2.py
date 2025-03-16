#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 2751                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/2751                           #+#        #+#      #+#     #
#    Solved: 2025/01/08 14:36:46 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
import sys
n = int(sys.stdin.readline())
numbers = [int(sys.stdin.readline()) for _ in range(n)]
numbers.sort()
for number in numbers:
    print(number)

