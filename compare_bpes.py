from difflib import SequenceMatcher
import sys

fname1 = sys.argv[1]
fname2 = sys.argv[2]

with open(fname1) as f1, open(fname2) as f2:
    for line_num, (l1, l2) in enumerate(zip(f1.readlines(), f2.readlines())):
        t1 = l1.split()
        t2 = l2.split()

        if not(t1) or not(t2):
            continue
        
        edits = [
            code for code in SequenceMatcher(a=t1, b=t2, autojunk=False).get_opcodes()
            if code[0] != 'equal'
        ]
        if len(edits) > 7:
            print(line_num)
            print('< ', l1)
            print('> ', l2)
            print()

