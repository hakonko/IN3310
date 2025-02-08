def broadcasting(m1, m2):
    while len(m1) < len(m2):
        m1.insert(0, 1)
    while len(m2) < len(m1):
        m2.insert(0, 1)

    form = []
    for i in range(len(m1)):
        if m1[i] == m2[i] or m1[i] == 1 or m2[i] == 1:
            form.append(max(m1[i], m2[i]))
        else:
            return "Incompatible shapes"
    
    return form

m1s = [[3, 1, 3], [4, 1], [3], [1, 4], [6, 3, 1, 7], [6, 3, 1, 7], [1, 2, 3, 1, 6], [2, 5, 1, 7]]
m2s = [[2, 3, 3], [3, 1, 1, 5], [3, 1, 1, 5], [7, 1], [2, 7], [2, 1, 7], [8, 1, 3, 2, 6], [9, 2, 3, 2, 1]]

if __name__ == '__main__':
    for i in range(len(m1s)):
        print(f"{m1s[i]} and {m2s[i]} -> {broadcasting(m1s[i], m2s[i])}")