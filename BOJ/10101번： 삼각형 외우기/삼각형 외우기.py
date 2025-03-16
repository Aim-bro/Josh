#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 10101                             :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/10101                          #+#        #+#      #+#     #
#    Solved: 2024/12/09 15:26:48 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
angles = []
for _ in range(3):
    angles.append(int(input()))

if sum(angles) == 180:
    if len(set(angles)) == 1:
        print('Equilateral')
    elif len(set(angles)) == 2:
        print('Isosceles')
    else:
        print('Scalene')
else:
    print("Error")