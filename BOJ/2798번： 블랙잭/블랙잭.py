#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 2798                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/2798                           #+#        #+#      #+#     #
#    Solved: 2024/12/23 09:27:33 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
N, M = map(int, input().split())
cards = list(map(int, input().split()))

max_sum = 0
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            sum = cards[i] + cards[j] + cards[k]
            if sum <= M and sum > max_sum:
                max_sum = sum
print(max_sum)
