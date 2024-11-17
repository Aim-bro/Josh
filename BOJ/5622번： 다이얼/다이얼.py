#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 5622                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/5622                           #+#        #+#      #+#     #
#    Solved: 2024/11/17 09:49:59 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
s = 0
for x in input():
    if x in 'ABC':
        s += 3
    elif x in 'DEF':
        s += 4
    elif x in 'GHI':
        s += 5
    elif x in 'JKL':
        s += 6
    elif x in 'MNO':
        s += 7
    elif x in 'PQRS':
        s += 8
    elif x in 'TUV':
        s += 9
    elif x in 'WXYZ':
        s += 10

print(s)    