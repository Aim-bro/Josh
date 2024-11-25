#  **************************************************************************  #
#                                                                              #
#                                                       :::    :::    :::      #
#    Problem Number: 10798                             :+:    :+:      :+:     #
#                                                     +:+    +:+        +:+    #
#    By: jjh970323 <boj.kr/u/jjh970323>              +#+    +#+          +#+   #
#                                                   +#+      +#+        +#+    #
#    https://boj.kr/10798                          #+#        #+#      #+#     #
#    Solved: 2024/11/20 14:03:30 by jjh970323     ###          ###   ##.kr     #
#                                                                              #
#  **************************************************************************  #
arr = []
max_len = 0

for x in range(5):
    word = input()
    arr.append(word)
    if len(word) > max_len:
        max_len = len(word)

cor = []
for i in range(max_len):
    column = []
    for word in arr:
        if i < len(word):
            column.append(word[i])
        else:
            column.append('')
    cor.append(column)

result = ''.join([''.join(x).strip() for x in cor])
print(result)

