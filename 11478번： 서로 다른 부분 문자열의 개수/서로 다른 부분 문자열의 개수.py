#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 11478                             :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/11478                          #+#        #+#      #+#     #
#    Solved: 2025/02/06 12:15:57 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
import sys
input = sys.stdin.readline

S = input().strip()
S_set = set([S[i:j] for i in range(len(S)) for j in range(i+1, len(S)+1)])
print(len(S_set))