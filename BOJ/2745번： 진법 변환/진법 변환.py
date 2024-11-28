#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 2745                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/2745                           #+#        #+#      #+#     #
#    Solved: 2024/11/21 16:36:20 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
num_list = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
N, B = input().split()
answer = 0
for i, num in enumerate(N[::-1]):
    answer += int(B) ** i * num_list.index(num)
print(answer)