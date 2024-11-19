#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 1316                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/1316                           #+#        #+#      #+#     #
#    Solved: 2024/11/19 11:09:49 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
N = int(input())
tot = 0
for x in range(N):
    word = input()
    if len(word) > 2:
        for y in range(0,len(word)-2):
            if word[y] != word[y+1]:
                if word[y] in word[y+2:]:
                    tot -= 1
                    break           
        tot += 1
    else:
        tot += 1
print(tot)