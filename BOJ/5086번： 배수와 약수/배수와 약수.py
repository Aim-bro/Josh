#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 5086                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/5086                           #+#        #+#      #+#     #
#    Solved: 2024/11/27 10:14:44 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
while True:
    a, b = map(int,input().split())
    if a == 0 and b == 0:
        break
    if a//b > 0:
        if a%b == 0:
            print('multiple')
        else:
            print('neither')
    else:
        if b%a == 0:
            print('factor')
        else:
            print('neither')