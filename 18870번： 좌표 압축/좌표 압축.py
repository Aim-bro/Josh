#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 18870                             :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/18870                          #+#        #+#      #+#     #
#    Solved: 2025/01/13 11:37:17 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
N = int(input())
li = list(map(int, input().split()))

set_li = sorted(set(li))
# for i in li:
#     print(set_li.index(i), end=' ')
dic = {value: idx for idx, value in enumerate(set_li)}
print(' '.join(str(dic[x]) for x in li))
print(dic[-9])