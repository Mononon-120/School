import sys
class Pushdown_Automaton:
    def __init__(self, states: list, inputalphabet: list, stackalphabet: list, transitions: dict, initialstates: list, acceptingstates: list):
        self.states = states
        self.inputalphabet = inputalphabet
        self.stackalphabet = stackalphabet
        self.transitions = transitions
        self.initialstates = initialstates[0]
        self.acceptingstates = acceptingstates
        self.stack = []
        if self.stack:
            self.stacktop = self.stack[-1]
        else:
            self.stacktop = None
        self.currentstate = self.initialstates
        self.action = ("pop", "push", "stay")
        self.end = False

    def processsuport(self, string):
        if string != None:
            if string not in self.inputalphabet:
                raise ValueError(f"Invalid character '{string}' in input string.")
        if (self.currentstate, string, self.stacktop) in self.transitions:
            nextstate, action, pushsymbol = self.transitions[(self.currentstate, string, self.stacktop)] 
            if action in self.action:
                if action == "pop":
                    pop = self.stack.pop()
                    print(f"pop: {pop}")
                    if self.stack:
                        self.stacktop = self.stack[-1]
                    else:
                        self.stacktop = None
                    print(f"Transition: {self.currentstate}, {string}, {self.stacktop} --> {nextstate}, Stack: {self.stack}")
                    self.currentstate = nextstate
                elif action == "push":
                    self.stack.append(pushsymbol)
                    print(f"push: {pushsymbol}")
                    if self.stack:
                        self.stacktop = self.stack[-1]
                    else:
                        self.stacktop = None
                    print(f"Transition: {self.currentstate}, {string}, {self.stacktop} --> {nextstate}, Stack: {self.stack}")
                    self.currentstate = nextstate
                elif action == "stay":
                    print(f"Transition: {self.currentstate}, {string}, {self.stacktop} --> {nextstate}, Stack: {self.stack}")
                    self.currentstate = nextstate
                    pass
                else:
                    raise ValueError(f"Invalid action '{action}'.")
            else:
                raise ValueError(f"Invalid action '{action}'.")
        else:
            print(f"No transition found for ({self.currentstate}, {string}, {self.stacktop}).")
            raise ValueError(f"Invalid action '{action}'.")


    def process(self, inputstrings):
        self.currentstate = self.initialstates
        self.stack = []
        if inputstrings == "":
            if self.stack:
                self.stacktop = self.stack[-1]
            else:
                self.stacktop = None
            if (self.currentstate, None, self.stacktop) in self.transitions:
                self.processsuport(None)
        print(f"Start at: {self.currentstate}")
        for string in inputstrings:
            if self.stack:
                stacktop = self.stack[-1]
            else:
                self.stacktop = None
            check = False
            print(f"{self.currentstate}")
            if check == False:
                if (self.currentstate, string, self.stacktop) in self.transitions:
                    self.processsuport(string)
                    check = True
            if (self.currentstate, None, self.stacktop) in self.transitions:
                self.processsuport(None)
            if check == False:
                if (self.currentstate, string, self.stacktop) in self.transitions:
                    self.processsuport(string)
                    check = True
        if self.currentstate in self.acceptingstates:
            print("accept")
        else:
            print("reject")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 pda.py <strings>")
        sys.exit(1)
    user_input = sys.argv[1]
    print(f"{user_input}")
    states = ["q0", "q1", "q2", "q3", "q4", "q5"]
    inputalphabet = ["0", "1"]
    stackalphabet = ["1", "0"]
    transitions = {
            ("q0", None, None): ("q1", "push", "1"),
            ("q1", "0", "1"): ("q2", "push", "0"),
            ("q1", "1", "1"): ("q5", "stay", None),
            ("q2", "0", "0"): ("q2", "push", "0"),
            ("q2", "1", "0"): ("q3", "pop", None),
            ("q3", "1", "0"): ("q3", "pop", None),
            ("q3", None, "1"): ("q4", "pop", None),
            ("q4", "1", None): ("q5", "stay", None),
            ("q4", "0", None): ("q5", "stay", None),
            ("q1", None, "1"): ("q4", "pop", None)
            }
    initialstates = ["q0"]
    acceptingstates = ["q1", "q4"]
    pa = Pushdown_Automaton(states, inputalphabet, stackalphabet, transitions, initialstates, acceptingstates)
    pa.process(user_input)

if __name__ == "__main__":
    main()

