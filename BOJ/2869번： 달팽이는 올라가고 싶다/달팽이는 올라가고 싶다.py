#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 2869                              :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/2869                           #+#        #+#      #+#     #
#    Solved: 2024/11/26 12:09:52 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
# A, B, V = map(int,input().split())
# day = 0
# day = (V-A)//(A-B)
# if (V-A) % (A-B) > 0:
#     day += 1
# day += 1
# print(day)


# 시간 오류
# while V>0:
#     time += 1
#     V -= A
#     if V <= 0:
#         break
#     else:
#         V += B







A, B, V = map(int,input().split())
V = V-A
time = 1
if V%(A-B) == 0:
    time += V//(A-B)
else:
    time += V//(A-B) + 1
    
print(time)
    















