#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 3009                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/3009                           #+#        #+#      #+#     #
#    Solved: 2024/12/05 21:04:03 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
x_p = []
y_p = []
for _ in range(3):
    x, y = map(int,input().split())
    if x in x_p:
        x_p.remove(x)
    else:
        x_p.append(x)
    
    if y in y_p:
        y_p.remove(y)
    else:
        y_p.append(y)    
print(' '.join([str(x) for x in x_p+y_p]))
