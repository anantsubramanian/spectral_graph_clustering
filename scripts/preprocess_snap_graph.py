import sys

if not len(sys.argv) == 3:
    print("Usage: python preprocess_snap_graph.py <input-file> <output-file>")
    exit()

with open(sys.argv[1]) as fin:
    with open(sys.argv[2], "w") as fout:
        idmap = {}
        g = {}
        for a in fin.readlines():
            if a[0] == "#":
                continue
            a = a.strip().split()
            a = [ int(x) for x in a ]
            if not a[0] in idmap:
                idmap[a[0]] = len(idmap)
            if not a[1] in idmap:
                idmap[a[1]] = len(idmap)
            a[0] = idmap[a[0]]
            a[1] = idmap[a[1]]
            if not a[0] in g:
                g[a[0]] = []
            if not a[1] in g:
                g[a[1]] = []
            g[a[0]].append(a[1])
            g[a[1]].append(a[0])
        
        for v in g:
            g[v] = set(g[v])
        m = str(sum([len(g[v]) for v in g]))
        n = str(max([max(g[v]) for v in g])+1)
        
        fout.write(n + " " + m + "\n")
        for v in sorted(g):
            for vj in sorted(g[v]):
                fout.write(str(v) + " " + str(vj) + " 1\n")
        fout.close()
    fin.close()

