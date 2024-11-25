#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 2738                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/2738                           #+#        #+#      #+#     #
#    Solved: 2024/11/20 09:21:45 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
N, M = map(int,input().split())
arr = []
arr2 = []
result = []
for x in range(N):
    arr.append([int(i) for i in input().split()])
for y in range(N):
    arr2.append([int(i) for i in input().split()])

for row1, row2 in zip(arr,arr2):
    result_row =[]
    for v1, v2 in zip(row1,row2):
        result_row.append(v1+v2)
    result.append(result_row)

for x in result:
    print(' '.join([str(y) for y in x]))