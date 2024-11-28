#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 1978                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/1978                           #+#        #+#      #+#     #
#    Solved: 2024/11/28 10:40:03 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
N = int(input())
arr = [int(x) for x in input().split()]
cnt = len(arr)
for x in range(len(arr)):
    if arr[x] == 1:
        cnt -= 1
    else:
        for y in range(2,arr[x]):
            if arr[x]%y == 0:
                cnt -= 1
                break

print(cnt)            