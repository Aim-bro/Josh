#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 10815                             :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/10815                          #+#        #+#      #+#     #
#    Solved: 2025/01/14 09:39:10 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
N = int(input())
li = list(map(int, input().split()))
li = set(li)
M = int(input())
li2 = list(map(int, input().split()))

print(' '.join(str(1) if x in li else str(0) for x in li2))

