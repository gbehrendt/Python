# ME:4150 Artificial Intelligence in Engineering
# hw4: CSPs
# by Prof. Shaoping Xiao


from cspProblemDefine import CSP, Constraint, ne_, is_ 
from operator import lt,ne,eq,gt
from search import Search_from_CSP, Searcher

def meet_at(p1,p2):
    """returns a function that is true when the words meet at the postions p1, p2
    """
    def meets(w1,w2):
        return w1[p1] == w2[p2]
    meets.__name__ = "meet_at("+str(p1)+','+str(p2)+')'
    return meets

crossword1 = CSP({'one_across':{'ant', 'bus', 'car', 'has'},
                  'one_down':{'buys', 'hold', 'lane', 'year'},
                  'three_across':{'buys', 'hold', 'lane', 'year'},
                  'two_down':{'search', 'syntax'},
                  'four_across':{'ant', 'bus', 'car', 'has'}
                  },
                  [Constraint(('one_across','one_down'),meet_at(0,0)),
                   Constraint(('one_down','three_across'),meet_at(2,0)),
                   Constraint(('one_across','two_down'),meet_at(2,0)),
                   Constraint(('three_across','two_down'),meet_at(2,2)),
                   Constraint(('four_across','two_down'),meet_at(0,4))
                   ])
    
searcher3 = Searcher(Search_from_CSP(crossword1))
print('The first solution searched is:')
print(searcher3.search())

