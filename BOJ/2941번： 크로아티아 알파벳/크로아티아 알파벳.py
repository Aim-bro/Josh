#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 2941                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/2941                           #+#        #+#      #+#     #
#    Solved: 2024/11/19 09:23:06 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
arr = ["c=", "c-", "dz=", "d-", "lj", "nj", "s=", "z="]

sen = input()
tot = 0
i = 0
while i < len(sen):
    match = False
    for word in arr:
        if sen[i:i+len(word)] == word:
            tot += 1
            i += len(word)
            match = True
            break
    if not match:
        tot += 1
        i += 1

print(tot)
    
