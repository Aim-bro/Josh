#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 9506                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/9506                           #+#        #+#      #+#     #
#    Solved: 2024/11/28 10:20:51 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
while True:
    N = int(input())
    if N == -1:
        break
    arr = []
    for x in range(1,N):
        if N%x == 0:
            arr.append(x)
    if sum(arr) == N:
        print(f'{N} = {" + ".join(map(str,arr))}')
    else:
        print(f'{N} is NOT perfect.')