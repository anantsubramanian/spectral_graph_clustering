import sys
import operator

if not len(sys.argv) == 3:
    print("Usage: python sort_by_dest.py <input-file> <output-file>")
    exit()

with open(sys.argv[1]) as fin:
    with open(sys.argv[2], "w") as fout:
        g = []
        for line in fin:
            g.append([ int(x) for x in line.strip().split() ])
        
        fout.write(str(g[0][0]) + " " + str(g[0][1]) + "\n")
        g = sorted(g[1:], key=operator.itemgetter(1))
        
        for vi,vj,w in g:
            fout.write(str(vi) + " " + str(vj) + " " + str(w) + "\n")
        
        fout.close()
    fin.close()

