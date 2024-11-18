#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 1157                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/1157                           #+#        #+#      #+#     #
#    Solved: 2024/11/18 22:58:41 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
word = input().upper()
li = {}
for x in word:
    if x in li:
        li[x] += 1
    else:
        li[x] = 1
        
max_val = max(li.values())
max_key = [key for key, value in li.items() if value == max_val]
if len(max_key) == 1:
    print(max_key[0])
else:
    print('?')