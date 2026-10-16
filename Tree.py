import random
import operator
import sympy as sp
import copy 
import math               

# binary operators and unary operators
OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
U_OPS = {"sin": sp.sin, "log": sp.log, "^": sp.exp}

# var names
VARS = ["x1", "x2", "x3"]            

class Tree:
    def __init__(self, data):
        self.data = data     
        self.left = None
        self.right = None
        self.repairs = 0  # keep track of manual overrides

    def is_leaf(self):
        if self.left == None and self.right == None:
            return True
        else:
            return False
        
    def calculate_value(self, x_list):   
        # x_list is one row of inputs, like [x1, x2, x3].
        # recursive: trickle down the operator nodes to the children, evaluates its children, then combines them                    
        if self.data == "*":
            return self.left.calculate_value(x_list) * self.right.calculate_value(x_list)
        if self.data == "/":
            num = self.left.calculate_value(x_list)
            den = self.right.calculate_value(x_list)
            if den == 0:         # protected division: x/0 returns x (Koza returns 1) so trees never crash
                self.repairs += 1
                return num       
            return num / den     
        if self.data == "-":
            return self.left.calculate_value(x_list) - self.right.calculate_value(x_list)
        if self.data == "+":
            return self.left.calculate_value(x_list) + self.right.calculate_value(x_list)
        if self.data in VARS:                            
            return x_list[VARS.index(self.data)]               
        if self.data == "sin":
            return math.sin(self.left.calculate_value(x_list))
        if self.data == "log":
            val = self.left.calculate_value(x_list)
            if val <= 0:         # protected log: about half of dataset 3's x values are <= 0, so GP can exploit this
                self.repairs += 1
                return 0.0       
            return math.log(val) 
        if self.data == "^":
            val = self.left.calculate_value(x_list)
            if val > 50:         # cap exp at e^50 to prevent overflow
                self.repairs += 1
                val = 50         
            return math.exp(val) 
        else:
            # nothing matched, so this must be a constant leaf so return the number itself
            return self.data
        
    def print_equation(self):
        equation = []
        if self.left is not None:
            equation += [self.left.print_equation()]
        equation.append(self.data)
        if self.right is not None:
            equation += [self.right.print_equation()]
        return equation   
    
    def height(self):
        if self.is_leaf():
            return 0
        left  = self.left.height() if self.left is not None else -1
        right = self.right.height() if self.right is not None else -1
        return max(left, right) + 1
    
    def to_list(self):
        nodes = [self]
        if self.left is not None:
            nodes += self.left.to_list()
        if self.right is not None:
            nodes += self.right.to_list()
        return nodes
    
    def random_node(self):
        return random.choice(self.to_list())
    
    
    def mutate(self, Multi=False):
        """
        Return a *new* Tree that is a mutated version of this tree.

        DO NOT MODIFY self!
        """
        if Multi:
            pool = VARS
        else: pool = ["x1"]
        child = copy.deepcopy(self)     # change a copy, never self (immutable, like Tour)
        node = child.random_node()      # point mutation: ops only swap with same-arity ops, so the tree stays valid
        if node.data in list(OPS):
            valid_ops = []
            for op in list(OPS):
                if node.data != op:
                    valid_ops.append(op)
            node.data = random.choice(valid_ops)
        elif node.data in list(U_OPS):
            valid_U_ops = []
            for uop in list(U_OPS):
                if node.data != uop:
                    valid_U_ops.append(uop)
            node.data = random.choice(valid_U_ops)
        elif node.data in VARS:
            if random.random() < .5 or len(pool) == 1:      # -> KNOB
                node.data = round(random.uniform(-5,5),2)
            else:
                valid_vars = []
                for v in pool:
                    if v != node.data:
                        valid_vars.append(v)
                node.data = random.choice(valid_vars)
        else:
            if random.random() < .4:                        # -> KNOB
                node.data = random.choice(pool)
            else:
                node.data = round(random.uniform(-5,5),2)
        return child
    
    def crossover(self, other):
        """
        Return a *new* Tree that is an offspring produced by
        crossing over this Tree with the other Tree.

        DO NOT MODIFY self or other!
        """
        child = copy.deepcopy(self)
        clist = child.to_list()
        cplist = []
        for node in clist:
            if not (node.is_leaf()):
                cplist.append(node)
        if len(cplist) == 0:
            return copy.deepcopy(other)
        crnode = random.choice(cplist)
        olist = other.to_list()
        oplist = []
        oclist = []
        for node in olist:
            if not (node.is_leaf()):
                oplist.append(node)
            else:
                oclist.append(node)
        # donor subtree is an internal node 75% of the time, so crossover mostly swaps real chunks, not single leaves
        if random.random() <.75 and len(oplist) != 0:
            ornode = random.choice(oplist)
        else:
            ornode = random.choice(oclist)
        ornode = copy.deepcopy(ornode)
        if random.random() <= .5:
            if crnode.left is not None:
                crnode.left = ornode
            else:
                crnode.right = ornode
        else: 
            if crnode.right is not None:
                crnode.right = ornode
            else:
                crnode.left = ornode
        return child
        
def random_tree(depth=3, Unary=False, Multi=False):
    # "grow" method: stop at max depth, or 30% of the time at any level, so tree shapes vary
    if depth == 0 or random.random() < 0.3:
        return random_leaf(Multi)
    if Unary == True and random.random() < .3:          # -> KNOB
        root = Tree(random.choice(list(U_OPS)))
        root.left = random_tree(depth - 1, Unary, Multi)
        return root
    root = Tree(random.choice(list(OPS)))
    root.left = random_tree(depth - 1, Unary, Multi)
    root.right = random_tree(depth - 1, Unary, Multi)
    return root
def random_leaf(Multi=False):
    if random.random() <= 0.5:
        if Multi:
            return Tree(random.choice(VARS))
        else:
            return Tree(VARS[0])
    return Tree(round(random.uniform(-5, 5), 2))
def print_equation_string(equation):
    def evaluate(node):
        if len(node) == 1:
            return sp.sympify(node[0])
        if len(node) == 2:
            child, op = node
            return U_OPS[op](evaluate(child))
        left, op, right = node
        return OPS[op](evaluate(left), evaluate(right))
    return str(sp.expand(evaluate(equation)))
def main():
    a = random_tree(Unary=True, Multi=True)
    b = random_tree(Unary=False, Multi=True)                
    c = a.crossover(b)
    print(a.print_equation())
    print(b.print_equation())
    print(c.print_equation())
    d = a.mutate(Multi=True)
    print(a.print_equation())
    print(d.print_equation())
if __name__ == "__main__":
    main()