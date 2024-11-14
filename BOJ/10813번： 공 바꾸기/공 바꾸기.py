#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 10813                             :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/10813                          #+#        #+#      #+#     #
#    Solved: 2024/11/14 13:31:03 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
N, M = map(int,input().split())
arr = [x+1 for x in range(N)]
for x in range(M):
    i, j = map(int, input().split())
    arr[i-1], arr[j-1] = arr[j-1], arr[i-1]
print(' '.join([str(x) for x in arr]))
