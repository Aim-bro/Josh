#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 2720                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/2720                           #+#        #+#      #+#     #
#    Solved: 2024/11/25 14:31:27 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
T = int(input())
Qu = 25
Di = 10
Ni = 5
Pe = 1
for x in range(T):
    C = int(input())
    C_Qu = C//Qu
    C = C%Qu
    C_Di = C//Di
    C = C%Di
    C_Ni = C//Ni
    C = C%Ni
    C_Pe = C//Pe
    C = C%Pe
    print(C_Qu, C_Di, C_Ni, C_Pe) 