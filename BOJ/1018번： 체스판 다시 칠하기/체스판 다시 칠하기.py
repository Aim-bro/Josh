#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 1018                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/1018                           #+#        #+#      #+#     #
#    Solved: 2024/12/31 23:28:53 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
M, N = map(int, input().split())
board = [input() for _ in range(M)]

list = []

for x in range(M-7):
    for y in range(N-7):
        count1 = 0
        count2 = 0
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:  # 시작점과 같은 색이어야 하는 위치
                    if board[x+i][y+j] != 'W':
                        count1 += 1
                    if board[x+i][y+j] != 'B':
                        count2 += 1
                else:  # 시작점과 다른 색이어야 하는 위치
                    if board[x+i][y+j] != 'B':
                        count1 += 1
                    if board[x+i][y+j] != 'W':
                        count2 += 1
        list.append(min(count1, count2))
print(min(list))