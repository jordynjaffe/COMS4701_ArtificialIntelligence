"""
Each futoshiki board is represented as a dictionary with string keys and
int values.
e.g. my_board['A1'] = 8

Empty values in the board are represented by 0

An * after the letter indicates the inequality between the row represented
by the letter and the next row.
e.g. my_board['A*1'] = '<' 
means the value at A1 must be less than the value
at B1

Similarly, an * after the number indicates the inequality between the
column represented by the number and the next column.
e.g. my_board['A1*'] = '>' 
means the value at A1 is greater than the value
at A2

Empty inequalities in the board are represented as '-'

"""
import sys
import time
import numpy as np
import copy
#======================================================================#
#*#*#*# Optional: Import any allowed libraries you may need here #*#*#*#
#======================================================================#

#=================================#
#*#*#*# Your code ends here #*#*#*#
#=================================#

ROW = "ABCDEFGHI"
COL = "123456789"

class Board:
    '''
    Class to represent a board, including its configuration, dimensions, and domains
    '''
    
    def get_board_dim(self, str_len):
        '''
        Returns the side length of the board given a particular input string length
        '''
        d = 4 + 12 * str_len
        n = (2+np.sqrt(4+12*str_len))/6
        if(int(n) != n):
            raise Exception("Invalid configuration string length")
        
        return int(n)
        
    def get_config_str(self):
        '''
        Returns the configuration string
        '''
        return self.config_str
        
    def get_config(self):
        '''
        Parses the configuration string and returns the configuration dictionary
        '''
        n = self.n
        config_str = self.config_str
        config_dict = {}
        #HARD CODE UPDATES TO CONFIG_STR
        grid_size = 2 * n - 1
        idx = 0  # Index to keep track of position in config_str
        for i in range(grid_size):
            for j in range(grid_size):
                if i % 2 == 0 and j % 2 == 0:
                    # Variable cell
                    row = self.ROW[i // 2]
                    col = self.COL[j // 2]
                    var_key = row + col
                    value = int(config_str[idx])
                    config_dict[var_key] = value
                    idx += 1
                elif i % 2 == 0 and j % 2 == 1:
                    # Horizontal inequality
                    row = self.ROW[i // 2]
                    col = self.COL[j // 2]
                    ineq_key = row + col + '*'
                    value = config_str[idx]
                    config_dict[ineq_key] = value
                    idx += 1
                elif i % 2 == 1 and j % 2 == 0:
                    # Vertical inequality
                    row = self.ROW[i // 2]
                    col = self.COL[j // 2]
                    ineq_key = row + '*' + col
                    value = config_str[idx]
                    config_dict[ineq_key] = value
                    idx += 1
                # Positions where both i and j are odd are skipped
        return config_dict

        
    def get_variables(self):
        '''
        Returns a list containing the names of all variables in the futoshiki board
        '''
        variables = []
        for i in range(0, self.n):
            for j in range(0, self.n):
                variables.append(ROW[i] + COL[j])
        return variables
    
    def convert_string_to_dict(self, config_string):
        '''
        Parses an input configuration string, retuns a dictionary to represent the board configuration
        as described above
        '''
        config_dict = {}
        
        for i in range(0, self.n):
            for j in range(0, self.n):
                cur = config_string[0]
                config_string = config_string[1:]
                
                config_dict[ROW[i] + COL[j]] = int(cur)
                
                if(j != self.n - 1):
                    cur = config_string[0]
                    config_string = config_string[1:]
                    config_dict[ROW[i] + COL[j] + '*'] = cur
                    
            if(i != self.n - 1):
                for j in range(0, self.n):
                    cur = config_string[0]
                    config_string = config_string[1:]
                    config_dict[ROW[i] + '*' + COL[j]] = cur
                    
        return config_dict
        
    def print_board(self):
        '''
        Prints the current board to stdout
        '''
        config_dict = self.config
        for i in range(0, self.n):
            for j in range(0, self.n):
                cur = config_dict[ROW[i] + COL[j]]
                if(cur == 0):
                    print('_', end=' ')
                else:
                    print(str(cur), end=' ')
                
                if(j != self.n - 1):
                    cur = config_dict[ROW[i] + COL[j] + '*']
                    if(cur == '-'):
                        print(' ', end=' ')
                    else:
                        print(cur, end=' ')
            print('')
            if(i != self.n - 1):
                for j in range(0, self.n):
                    cur = config_dict[ROW[i] + '*' + COL[j]]
                    if(cur == '-'):
                        print(' ', end='   ')
                    else:
                        print(cur, end='   ')
            print('')
    
    # def __init__(self, config_string):
    #     '''
    #     Initialising the board
    #     '''
    #     self.config_str = config_string
    #     self.n = self.get_board_dim(len(config_string))
    #     if(self.n > 9):
    #         raise Exception("Board too big")
    #
    #     self.config = self.convert_string_to_dict(config_string)
    #     self.domains = self.reset_domains()
    #
    #     self.forward_checking(self.get_variables())

    def __init__(self, config_string):
        '''
        Initializing the board
        '''
        self.config_str = config_string
        self.n = self.get_board_dim(len(config_string))
        if self.n > 9:
            raise Exception("Board too big")

        # Generate ROW and COL based on board size
        self.ROW = [chr(ord('A') + i) for i in range(self.n)]
        self.COL = [str(i + 1) for i in range(self.n)]

        self.config = self.get_config()
        self.variables = self.get_variables()
        self.domains = self.reset_domains()
        self.inequalities = self.find_inequalities()

        # Initial forward checking based on initial assignments
        #different from later forward checking within backtracking
        self.initial_forward_checking()

    def __str__(self):
        '''
        Returns a string displaying the board in a visual format. Same format as print_board()
        '''
        output = ''
        config_dict = self.config
        for i in range(0, self.n):
            for j in range(0, self.n):
                cur = config_dict[ROW[i] + COL[j]]
                if(cur == 0):
                    output += '_ '
                else:
                    output += str(cur)+ ' '
                
                if(j != self.n - 1):
                    cur = config_dict[ROW[i] + COL[j] + '*']
                    if(cur == '-'):
                        output += '  '
                    else:
                        output += cur + ' '
            output += '\n'
            if(i != self.n - 1):
                for j in range(0, self.n):
                    cur = config_dict[ROW[i] + '*' + COL[j]]
                    if(cur == '-'):
                        output += '    '
                    else:
                        output += cur + '   '
            output += '\n'
        return output
        
    def reset_domains(self):
        '''
        Resets the domains of the board assuming no enforcement of constraints
        '''
        domains = {}
        variables = self.get_variables()
        for var in variables:
            if(self.config[var] == 0):
                domains[var] = [i for i in range(1,self.n+1)]
            else:
                domains[var] = [self.config[var]]
                
        self.domains = domains
                
        return domains

    # def forward_checking(self, reassigned_variables):
    #     '''
    #     Runs the forward checking algorithm to restrict the domains of all variables based on the values
    #     of reassigned variables
    #     '''
    #
    #     # Create a queue of arcs based on reassigned variables and their neighbors
    #     queue = [(var, neighbor) for var in reassigned_variables for neighbor in self.get_neighbors(var)]
    #     # Debug
    #     print("Initial Queue:", queue)
    #     print("Domains:", self.domains)
    #
    #     # While the queue is not empty
    #     while queue:
    #         Xi, Xj = queue.pop(0)  # Get the first arc
    #         print(f"Processing arc: {Xi} -> {Xj}")
    #
    #         # Retrieve the list of inequalities that apply to this arc
    #
    #         # If revise(csp, xi, xj) with inequalities returns True
    #         if self.revise(Xi, Xj):
    #             # If the domain of Xi is empty, return False (inconsistency found)
    #             if not self.domains[Xi]:
    #                 return False
    #
    #             # For each Xk in neighbors of Xi excluding Xj, add (Xk, Xi) to the queue
    #             for Xk in self.get_neighbors(Xi):
    #                 if Xk != Xj and (Xk, Xi) not in queue:
    #                     queue.append((Xk, Xi))
    #
    #     return True

    def initial_forward_checking(self):

        assigned_vars = [var for var in self.variables if len(self.domains[var]) == 1]
        #using initial domain, set up forward tracking
        #later we have deep copy --> use other forward_checking
        for var in assigned_vars:
            self.forward_check(var)

    #hard code find inequalities at start, that way don't have to worry about the is_below or is_right
    def find_inequalities(self):

        inequalities = []
        for key, value in self.config.items():
            if '*' in key and value in ['<', '>']:
                if key[-1] == '*':
                    # Horizontal
                    var1 = key[:-1]
                    col_index = self.COL.index(var1[1])
                    var2 = var1[0] + self.COL[col_index + 1]
                else:
                    # Vertical
                    var1 = key.replace('*', '')
                    row_index = self.ROW.index(var1[0])
                    var2 = self.ROW[row_index + 1] + var1[1]
                inequalities.append((var1, var2, value))
                #print(inequalities)
        return inequalities

    def forward_check(self, var):

        value = self.domains[var][0]

        for neighbor in self.find_neighbors(var):
            if value in self.domains[neighbor]:
                self.domains[neighbor].remove(value)
                if not self.domains[neighbor]:
                    return False
        # inequality constraints
        for (var1, var2, ineq) in self.inequalities:
            if var == var1 and len(self.domains[var1]) == 1:
                if not self.enforce_inequality(var1, var2, ineq):
                    return False
            elif var == var2 and len(self.domains[var2]) == 1:
                if not self.enforce_inequality(var1, var2, ineq):
                    return False
        return True

    def find_neighbors(self, var):
        neighbors = []
        #define neighbors as same row/column and adjacent
        row, col = var[0], var[1]
        for v in self.variables:
            if v != var and (v[0] == row or v[1] == col):
                neighbors.append(v)
        return neighbors

    def enforce_inequality(self, var1, var2, ineq):
        val1 = self.domains[var1][0]
        if len(self.domains[var2]) > 1:
            if ineq == '<':
                self.domains[var2] = [v for v in self.domains[var2] if v > val1]
            elif ineq == '>':
                self.domains[var2] = [v for v in self.domains[var2] if v < val1]
            if not self.domains[var2]:
                return False
        return True

    def is_consistent(self, var, value, assignment):
        '''
        Check if assign value to var is consistent with the current assignment
        '''

        row, col = var[0], var[1]
        for v in assignment:
            if assignment[v] == value:
                if v[0] == row or v[1] == col:
                    return False
        # Check ineq constraints
        for (var1, var2, ineq) in self.inequalities:
            if var == var1 and var2 in assignment:
                if not self.check_inequality(value, assignment[var2], ineq):
                    return False
            if var == var2 and var1 in assignment:
                if not self.check_inequality(assignment[var1], value, ineq):
                    return False
        return True

    def check_inequality(self, val1, val2, ineq):
    #define operators-> running into errors in later backtrack of big boards!!! CHECK THIS!
        if ineq == '<':
            return val1 < val2
        elif ineq == '>':
            return val1 > val2
        else:
            return True

    def select_unassigned_variable(self, assignment):
        '''
        Selects the next variable to assign using the MRV heuristic
        '''
        unassigned_vars = [v for v in self.variables if v not in assignment]
        # Minimum Remaining Values heuristic
        min_domain_size = min(len(self.domains[v]) for v in unassigned_vars)
        # Select variables with the smallest domain
        candidates = [v for v in unassigned_vars if len(self.domains[v]) == min_domain_size]
        return candidates[0]


    def assign(self, var, value, assignment):
        '''
        Assigns value to var and updates the domains
        '''
        assignment[var] = value
        self.domains[var] = [value]
        self.config[var] = value

    def unassign(self, var, assignment, saved_domains):
        '''
        Reverts the assignment of var and restores domains
        '''
        if var in assignment:
            del assignment[var]
        self.domains = saved_domains
        self.config[var] = 0

    def order_domain_values_MRV(self, var):
        return sorted(self.domains[var])

    def get_config_str(self):
        '''
        Returns the configuration string
        '''
        n = self.n
        config = self.config
        grid_size = 2 * n - 1
        config_str_list = []
        for i in range(grid_size):
            for j in range(grid_size):
                if i % 2 == 0 and j % 2 == 0:
                    # Variable cell
                    row = self.ROW[i // 2]
                    col = self.COL[j // 2]
                    var_key = row + col
                    config_str_list.append(str(config[var_key]))
                elif i % 2 == 0 and j % 2 == 1:
                    # Horizontal inequality
                    row = self.ROW[i // 2]
                    col = self.COL[j // 2]
                    ineq_key = row + col + '*'
                    config_str_list.append(config[ineq_key])
                elif i % 2 == 1 and j % 2 == 0:
                    # Vertical inequality
                    row = self.ROW[i // 2]
                    col = self.COL[j // 2]
                    ineq_key = row + '*' + col
                    config_str_list.append(config[ineq_key])
                # Positions where both i and j are odd are ignored
        return ''.join(config_str_list)

    def __str__(self):
        '''
        Returns a string displaying the board in a visual format
        '''
        output = ''
        n = self.n
        config = self.config
        grid_size = 2 * n - 1
        for i in range(grid_size):
            line = ''
            for j in range(grid_size):
                if i % 2 == 0 and j % 2 == 0:
                    # Variable cell
                    row = self.ROW[i // 2]
                    col = self.COL[j // 2]
                    var_key = row + col
                    val = config[var_key]
                    line += str(val) if val != 0 else '_'
                elif i % 2 == 0 and j % 2 == 1:
                    # Horizontal inequality
                    row = self.ROW[i // 2]
                    col = self.COL[j // 2]
                    ineq_key = row + col + '*'
                    val = config[ineq_key]
                    line += val if val != '-' else ' '
                elif i % 2 == 1 and j % 2 == 0:
                    # Vertical inequality
                    row = self.ROW[i // 2]
                    col = self.COL[j // 2]
                    ineq_key = row + '*' + col
                    val = config[ineq_key]
                    line += val if val != '-' else ' '
                else:
                    line += ' '
            output += line + '\n'
        return output.strip()

def backtracking_search(board):
    '''
    Performs backtracking search to solve the board
    '''

    def backtrack(assignment):
        if len(assignment) == len(board.variables):
            return assignment
        var = board.select_unassigned_variable(assignment)
        #pseudocode: use MRV heauristic
        for value in board.order_domain_values_MRV(var):
            if board.is_consistent(var, value, assignment):
                saved_domains = copy.deepcopy(board.domains)
                #print("check saved_domain", saved_domains)
                board.assign(var, value, assignment)
                inference = board.forward_check(var)
                if inference:
                    result = backtrack(assignment)
                    #print("hit backtrack if inference")
                    if result:
                        return result
                board.unassign(var, assignment, saved_domains)
        return None

    return backtrack({})


def solve_board(board):
    '''
    Solves the board using backtracking search with forward checking
    Returns the solved board and runtime
    '''
    start_time = time.time()
    solution = backtracking_search(board)
    #print("hit backtrack-solution)
    end_time = time.time()
    runtime = end_time - start_time
    if solution is not None:
        for var in solution:
            board.config[var] = solution[var]
    else:
        #NOT TESTING bad boards-> don't need to test this!
        print("No solution found.")
    return board, runtime

    # def forward_checking(self, reassigned_variables):
    #     '''
    #     Runs the forward checking algorithm to restrict the domains of all variables based on the values
    #     of reassigned variables
    #     '''
    #
    #     # Create a queue of arcs based on reassigned variables and their neighbors
    #     queue = [(var, neighbor) for var in reassigned_variables for neighbor in self.get_neighbors(var)]
    #     print("Initial Queue:", queue)
    #     print("Domains:", self.domains)
    #
    #     # While the queue is not empty
    #     while queue:
    #         Xi, Xj = queue.pop(0)  # Get the first arc
    #         print(f"Processing arc: {Xi} -> {Xj}")
    #
    #         # Revise the domain of Xi with respect to Xj
    #         if self.revise(Xi, Xj):
    #             if not self.domains[Xi]:
    #                 return False  # Inconsistency found
    #
    #             # For each Xk in neighbors of Xi excluding Xj, add (Xk, Xi) to the queue
    #             for Xk in self.get_neighbors(Xi):
    #                 if Xk != Xj and (Xk, Xi) not in queue:
    #                     queue.append((Xk, Xi))
    #
    #         # Now also check the reverse arc: Xj -> Xi
    #         if self.revise(Xj, Xi):
    #             if not self.domains[Xj]:
    #                 return False  # Inconsistency found
    #
    #             # For each Xk in neighbors of Xj excluding Xi, add (Xk, Xj) to the queue
    #             for Xk in self.get_neighbors(Xj):
    #                 if Xk != Xi and (Xk, Xj) not in queue:
    #                     queue.append((Xk, Xj))
    #
    #     return True

#need revise function for AC-3 forward checking
#returns True iff we revise the domain of Xi
#revised = False
#for each x in Di do
#if no value y in Dj allows (x, ) to satisfy the constraint between X; and X; then delete x from Di and revised = True
#return revised
    def revise(self, x, y):
        revised = False
        if x not in self.domains or y not in self.domains:
            return revised
        for value in self.domains[x][:]:  # Iterate over a copy of the domain
            # Check if there's no possible value in y's domain that satisfies the constraint
            if not any(self.satisfies_constraint(value, val, x, y) for val in self.domains[y]):
                self.domains[x].remove(value)
                print("removing ", value)
                print("from", self.domains[x])# Remove the value if it violates the constraint
                revised = True
        return revised


#helper function to get neighbors of a variable
    #
    #
    def satisfies_constraint(self, x, y, var1, var2):
        # Map rows (letters) to indices
        board_size = len(self.config)
        row_map = {chr(ord('A') + i): i for i in range(board_size)}

        row1, col1 = row_map[var1[0]], int(var1[1]) - 1
        row2, col2 = row_map[var2[0]], int(var2[1]) - 1

        # Check if var1 is to the right or below var2
        is_right = (row1 == row2 and col2 == col1 + 1)
        is_left = (row1 == row2 and col2 == col1 - 1)
        is_below = (row2 == row1 + 1 and col1 == col2)
        is_above = (row1 == row2 + 1 and col2 == col1)

        # Determine which variable is x and which is y based on position
        if is_right or is_below:
            temp_var1, temp_var2 = var1, var2
            temp_x, temp_y = x, y
        elif is_left or is_above:
            temp_var1, temp_var2 = var2, var1
            temp_x, temp_y = y, x
        #else:
            #return True  # Not in a directly comparable position
        is_right = (temp_x == temp_y and temp_var2 == temp_var1 + 1)
        #is_left = (temp_x == temp_y and temp_var2 == temp_var1 - 1)
        is_below = (temp_var2 == temp_var1 + 1 and temp_x == temp_y)
        #is_above = (temp_var1 == temp_var2 + 1 and temp_x == temp_y)
        # Check if temp_x and temp_y are assigned (not None)
        if temp_x is not None and temp_y is not None:
            # Check for same value constraint in the same row
            if row1 == row2 and temp_x == temp_y:
                print(
                    f"Constraint violated: {temp_x} == {temp_y} in the same row {temp_var1}, expected different values for row constraint")
                return False

            # Check for same value constraint in the same column
            if col1 == col2 and temp_x == temp_y:
                print(
                    f"Constraint violated: {temp_x} == {temp_y} in the same column {col1}, expected different values for column constraint")
                return False

            # Row inequality checks
            row_constraint_key = f"{temp_var1[0]}*{temp_var1[1]}"
            if row_constraint_key in self.config and is_below:
                constraint = self.config[row_constraint_key]
                if constraint == '-':
                    return True
                print(f"Checking row constraint between {temp_var1} and {temp_var2}: {temp_x} {constraint} {temp_y}")
                if constraint == '<' and not (temp_x < temp_y):
                    print(f"Constraint violated less than: Expected {temp_x} < {temp_y}")
                    return False
                elif constraint == '>' and not (temp_x > temp_y):
                    print(f"Constraint violated greater than: Expected {temp_x} > {temp_y}")
                    return False

            # Column inequality checks
            col_constraint_key = f"{temp_var1[0]}{temp_var1[1]}*"
            if col_constraint_key in self.config and is_right:
                constraint = self.config[col_constraint_key]
                if constraint == '-':
                    return True
                print(f"Checking column constraint between {temp_var1} and {temp_var2}: {temp_x} {constraint} {temp_y}")
                if constraint == '<' and not (temp_x < temp_y):
                    print(f"Constraint violated less than: Expected {temp_x} < {temp_y}")
                    return False
                elif constraint == '>' and not (temp_x > temp_y):
                    print(f"Constraint violated greater than: Expected {temp_x} > {temp_y}")
                    return False

        return True  # No violations found WOERKING NUMBER 1

    #helper function satisfies_constraints
    #FIXED BACKTRACKING TO BE ON BOARD!!! NOT SELF as seen below
    # def backtracking(self):
    #     '''
    #     Performs the backtracking algorithm to solve the board
    #     Returns only a solved board
    #
    #     #return self.backtrack({}) # Replace with return values [from lecture pseudocode]
    #     return self if self.backtrack({}) else None
    #         #=================================#
    #         #*#*#*# Your code ends here #*#*#*#
    #         #=================================#
    #
    #     #main backtracking function from lecture pseudo code:
    #
    # def update_config_str(self, var, value):
    #     '''
    #     Updates the config_str based on the assigned variable and its new value.
    #     '''
    #     # Convert the config_str to a list for mutability
    #     config_list = list(self.config_str)
    #
    #     # Get the index of the variable in the configuration string
    #     index = self.get_variable_index(var)  # Define this method to find the variable's index
    #
    #     if index is not None:
    #         # Update the corresponding index with the new value
    #         config_list[index] = str(value)
    #
    #         # Join the list back into a string and update config_str
    #         self.config_str = ''.join(config_list)
    #         print(f"Updated config_str: {self.config_str}")
    #     else:
    #         print(f"Variable '{var}' not found in config_str.")
    #
    # def get_variable_index(self, var):
    #     '''
    #     Returns the index of the variable in the config_str list.
    #     '''
    #     idx = 0
    #     for i in range(self.n):
    #         for j in range(self.n):
    #             cur_var = ROW[i] + COL[j]
    #             if cur_var == var:
    #                 return idx
    #
    #             idx += 1
    #
    #             # Account for inequalities in the string
    #             if j < self.n - 1:
    #                 idx += 1  # Skip inequality character
    #
    #         # Account for row inequalities
    #         if i < self.n - 1:
    #             idx += self.n  # Skip the entire row of inequalities
    #
    #     return None  # Return None if the variable is not found
    #
    # def backtrack(self, assignment):
    #     if self.is_complete(assignment):
    #         for var, value in assignment.items():
    #             self.config[var] = value
    #             self.update_config_str(var, value)
    #             #print(self.config[var])
    #         #self.get_config()
    #         print(f"Completed assignment: {assignment}. Final configuration: {self.config_str}")
    #         return assignment
    #
    #     var = self.select_unassigned_variable(assignment)
    #     print("hit var =", var)
    #     print(self.domains)
    #     for value in self.order_domain_values(var, assignment):
    #         print(f"Trying {var} = {value} with current assignment: {assignment}")
    #         if self.is_consistent(assignment, var, value):
    #             # Assign the value tentatively
    #             assignment[var] = value
    #             original_domains = {v: list(self.domains[v]) for v in self.domains}
    #
    #             print(f"Tentatively assigned {var} = {value}. Current assignment: {assignment}")
    #
    #             # Forward check
    #             if self.forward_checking([var]):
    #                 if self.is_consistent(assignment, var, value):
    #                     print(f"Checking forward with {var} = {value}. Domains: {self.domains}")
    #                     result = self.backtrack(assignment)
    #                     if result is not None:
    #                         print(f"Assigned {var} = {value}")
    #                         return result
    #
    #             # If forward check fails, backtrack and reset domains
    #             del assignment[var]
    #             self.domains = original_domains  # Restore domains after backtrack
    #             print(f"Backtracking from {var} = {value}")
    #     print("hit none")
    #     return None
    #
    #
    #
    #
    # # def select_unassigned_variable(self, assignment):
    # #     unassigned_vars = [var for var in self.get_variables() if var not in assignment]
    # #     #MRV
    # #     return min(unassigned_vars, key=lambda var: len(self.domains[var]), default=None)
    # def select_unassigned_variable(self, assignment):
    #     # Get the list of unassigned variables
    #     unassigned_vars = [var for var in self.get_variables() if var not in assignment]
    #
    #     # If no unassigned variables, return None
    #     if not unassigned_vars:
    #         return None
    #
    #     # MRV: Select the variable with the smallest domain size
    #     # If there are ties, we could also consider additional heuristics
    #     # (e.g., choosing the variable with the most constraints, etc.)
    #     mrv_var = min(unassigned_vars, key=lambda var: len(self.domains[var]))
    #
    #     return mrv_var
    #
    # def order_domain_values(self, var, assignment):
    #     print("order domain values hit")
    #     return self.domains[var]
    #
    # def is_complete(self, assignment):
    #is_complete used in other backtracking that wasn't self implemented; NEED BOARD IMPLEMENTATION!!
    #     # Check if all variables have been assigned
    #     if len(assignment) != len(self.get_variables()):
    #         return False
    #     return True
    #
    # def is_consistent(self, assignment, var, value):
    #     temp_assignment = assignment.copy()
    #     temp_assignment[var] = value
    #
    #     # Check against neighbors (for constraints)
    #     for neighbor in self.get_neighbors(var):
    #         if neighbor in temp_assignment:
    #             if not self.satisfies_constraint(value, temp_assignment[neighbor], var, neighbor):
    #                 print("hit satisfies constraint")
    #                 print(
    #                     f"Inconsistent assignment found: {var}={value} conflicts with {neighbor}={temp_assignment[neighbor]}")
    #                 return False
    #
    #     # Check for duplicates in the same row and column
    #     row, col = self.get_row_col(var)
    #     for other_var, other_value in temp_assignment.items():
    #         if other_var != var:  # Skip checking against itself
    #             other_row, other_col = self.get_row_col(other_var)
    #             # Ensure they are in the same row or column
    #             if other_row == row or other_col == col:
    #                 if other_value == value:
    #                     print(f"Inconsistent assignment found: {var}={value} duplicates with {other_var}={other_value}")
    #                     return False
    #
    #     # Check inequality constraints using the revise logic
    #     for neighbor in self.get_neighbors(var):
    #         if neighbor in temp_assignment:
    #             if not self.satisfies_constraint(value, temp_assignment[neighbor], var, neighbor):
    #                 print(
    #                     f"Inconsistent assignment found: {var}={value} violates inequality with {neighbor}={temp_assignment[neighbor]}")
    #                 return False
    #
    #     return True
    #
    def get_row_col(self, var):
        # Assuming var is in the format 'A1', 'B2', etc.
        row = int(var[1]) - 1  # Adjust index for zero-based
        col = ord(var[0]) - ord('A')  # Convert letter to index (A=0, B=1, ...)
        return row, col

def print_stats(runtimes):
    '''
    Prints a statistical summary of the runtimes of all the boards
    '''
    min = 100000000000
    max = 0
    sum = 0
    n = len(runtimes)

    for runtime in runtimes:
        sum += runtime
        if(runtime < min):
            min = runtime
        if(runtime > max):
            max = runtime

    mean = sum/n

    sum_diff_squared = 0

    for runtime in runtimes:
        sum_diff_squared += (runtime-mean)*(runtime-mean)

    std_dev = np.sqrt(sum_diff_squared/n)

    print("\nRuntime Statistics:")
    print("Number of Boards = {:d}".format(n))
    print("Min Runtime = {:.8f}".format(min))
    print("Max Runtime = {:.8f}".format(max))
    print("Mean Runtime = {:.8f}".format(mean))
    print("Standard Deviation of Runtime = {:.8f}".format(std_dev))
    print("Total Runtime = {:.8f}".format(sum))


if __name__ == '__main__':
    if len(sys.argv) > 1:

        # Running futoshiki solver with one board $python3 futoshiki.py <input_string>.
        print("\nInput String:")
        print(sys.argv[1])
        
        print("\nFormatted Input Board:")
        board = Board(sys.argv[1])
        board.print_board()
        
        solved_board, runtime = solve_board(board)
        
        print("\nSolved String:")
        print(solved_board.get_config_str())
        
        print("\nFormatted Solved Board:")
        solved_board.print_board()
        
        print_stats([runtime])

        # Write board to file
        out_filename = 'output.txt'
        outfile = open(out_filename, "w")
        outfile.write(solved_board.get_config_str())
        outfile.write('\n')
        outfile.close()

    else:
        # Running futoshiki solver for boards in futoshiki_start.txt $python3 futoshiki.py

        #  Read boards from source.
        src_filename = 'futoshiki_start.txt'
        try:
            srcfile = open(src_filename, "r")
            futoshiki_list = srcfile.read()
            srcfile.close()
        except:
            print("Error reading the sudoku file %s" % src_filename)
            exit()

        # Setup output file
        out_filename = 'output.txt'
        outfile = open(out_filename, "w")
        
        runtimes = []

        # Solve each board using backtracking
        for line in futoshiki_list.split("\n"):
            
            print("\nInput String:")
            print(line)
            
            print("\nFormatted Input Board:")
            board = Board(line)
            board.print_board()
            
            solved_board, runtime = solve_board(board)
            runtimes.append(runtime)
            
            print("\nSolved String:")
            print(solved_board.get_config_str())
            
            print("\nFormatted Solved Board:")
            solved_board.print_board()

            # Write board to file
            outfile.write(solved_board.get_config_str())
            outfile.write('\n')

        # Timing Runs
        print_stats(runtimes)
        
        outfile.close()
        print("\nFinished all boards in file.\n")
