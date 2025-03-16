#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 7785                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/7785                           #+#        #+#      #+#     #
#    Solved: 2025/01/24 10:28:54 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
N = int(input())
li = set()
for _ in range(N):
    name, stat = input().split()
    if stat == "enter":
        li.add(name)
    else:
        li.remove(name)

for name in sorted(li, reverse=True):
    print(name)
