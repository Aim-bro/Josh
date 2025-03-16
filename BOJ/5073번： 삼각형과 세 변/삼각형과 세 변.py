#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 5073                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/5073                           #+#        #+#      #+#     #
#    Solved: 2024/12/11 09:57:51 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
while True:
    a, b, c = map(int,input().split())
    li = [a,b,c]
    if sum(li) == 0:
        break
    if max(li) >= sum(li)-max(li):
        print('Invalid')
    else:
        if len(set(li)) == 3:
            print('Scalene')
        elif len(set(li)) == 2:
            print('Isosceles')
        else:
            print('Equilateral')
    
