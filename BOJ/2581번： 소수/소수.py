#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 2581                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/2581                           #+#        #+#      #+#     #
#    Solved: 2024/11/29 09:32:04 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
# M = int(input())
# N = int(input())
# arr=[x for x in range(M,N+1)]
# for num in range(M,N+1):
#     if num > 1:
#         for x in range(2,int(num**0.5)+1):
#             if num%x == 0:
#                 arr.remove(num)
#                 break
# if len(arr) == 0:
#     print(-1)
# else:
#     print(sum(arr))
#     print(arr[0])

M = int(input())
N = int(input())

arr = []

for num in range(M, N + 1):
    if num > 1:
        prime = True
        for x in range(2, int(num**0.5) + 1):
            if num % x == 0:
                prime = False
                break
        if prime:
            arr.append(num)

if not arr:
    print(-1)
else:
    print(sum(arr))
    print(min(arr))
