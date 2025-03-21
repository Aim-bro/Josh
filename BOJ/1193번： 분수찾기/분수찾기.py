#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 1193                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/1193                           #+#        #+#      #+#     #
#    Solved: 2024/11/26 11:30:14 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
# 1  2  6  7 15
# 3  5  8  14
# 4  9  13
# 10 12 
# 11

# 1 2~3 4~6 7~10 11~15

# 1,3,6,10,15 (n*(n+1))/2
# 2,3,4,5
X = int(input())
li = 1
while X > li:
    X -= li
    li += 1

if li % 2 == 0:
    a = X
    b = li - X + 1
else:
    a = li - X + 1 
    b = X

print(f'{a}/{b}')