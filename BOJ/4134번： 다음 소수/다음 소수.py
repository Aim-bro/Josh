#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 4134                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/4134                           #+#        #+#      #+#     #
#    Solved: 2025/02/17 09:12:48 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
import sys
input = sys.stdin.readline
for _ in range(int(input())):
    n = int(input())
    while True:
        is_prime = True
        
        if n < 2:
            is_prime = False
        else:
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    is_prime = False
                    break
        
        if is_prime:
            print(n)
            break
        n += 1
                

