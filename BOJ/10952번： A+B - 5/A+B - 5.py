#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 10952                             :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/10952                          #+#        #+#      #+#     #
#    Solved: 2024/11/13 08:54:59 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
while True:
    a, b = map(int,input().split())
    if a == 0 and b == 0:
        break
    else:
        print(a+b)