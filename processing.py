filename = 'data/YAGO-WIKI20K/'

links = []
train_link = 0
lessPair = True
unsup = True
E = [dict(), dict(), dict()]
for line in open(filename + 'ref_pairs', 'r'):
    e1, e2 = map(int, line.split())
    links.append((e1, e2))
if unsup:
    for line in open(filename + 'sup_pairs', 'r'):
        e1, e2 = map(int, line.split())
        links.append((e1, e2))
elif lessPair:
    for line in open(filename + 'sup_pairs', 'r'):
        e1, e2 = map(int, line.split())
        train_link += 1
        if train_link >= 400:
            links.append((e1, e2))

for i in range(30000):
    E[1][i] = [0, 0]
    E[2][i + 15000] = [0, 0]
for i in [1, 2]:
    for line in open(filename + 'triples_' + str(i), 'r'):
        words = line.split()
        head, r, tail, t1, t2 = [int(item) for item in words]

        E[i][head][1] += 2
        E[i][tail][1] += 2
        if t1 > 0:
            E[i][head][0] += 1
            E[i][tail][0] += 1
        if t2 > 0:
            E[i][head][0] += 1
            E[i][tail][0] += 1

def get_score(e, x)->bool:
    c0 = E[x][e][0]
    c1 = E[x][e][1]
    if c1 > 0 and c0/c1 > 0.5:
        return True
    return False
if unsup:
    addition = '2'
elif lessPair:
    addition = '1'
else:
    addition = ''
f1 = open(filename + 'time_sensitive_link' + addition, 'w')
f2 = open(filename + 'not_sensitive_link' + addition, 'w')
cnt1, cnt2 = 0, 0
for e1, e2 in links:
    s1 = get_score(e1, 1)
    s2 = get_score(e2, 2)
    if s1 and s2:
        f1.write(str(e1) + '\t' + str(e2) + '\n')
        cnt1 += 1
    else:
        f2.write(str(e1) + '\t' + str(e2) + '\n')
        cnt2 += 1
print(cnt1, cnt2, cnt1/cnt2)

