# ME:4150 Artificial Intelligence in Engineering
# hw2: Search by DFS
# by Prof. Shaoping Xiao

# you need copy searchProb.py and searchDFS.py in the same folder

from searchProb import Arc, Search_problem_graph, Search_problem
from searchDFS import SearchDFS

Rob_deliver = Search_problem_graph(
    {'mail','ts','o103','o109','o111','b1','b2','b3','b4','c1','c2','c3',
     'o125','o123','o119','r123','storage'},
     [Arc('o103','ts',8),
        Arc('ts','mail',6),
        Arc('o103','b3',4),
        Arc('b3','b1',4),
        Arc('b3','b4',7),
        Arc('b1','b2',6),
        Arc('b2','b4',3),
        Arc('b4','o109',7),
        Arc('b1','c2',3),
        Arc('c2','c3',6),
        Arc('c2','c1',4),
        Arc('c1','c3',8),
        Arc('o103','o109',12),
        Arc('o109','o111',4),
        Arc('o109','o119',16),
        Arc('o119','o123',9),
        Arc('o119','storage',7),
        Arc('o123','o125',4),
        Arc('o123','r123',4),
        ],
    start = 'o103',
    goals = {'r123'},
    )

searcher1 = SearchDFS(Rob_deliver)   # DFS
print(searcher1.search())  # find first path
print(searcher1.search())  # find next path
print(searcher1.search())   # find the third path



