class temp():
    def __init__(self,state):
        self.state = state
    
    def add(self):
        self.state.append(1)


list1=[1,2,3]
list2 = tuple(list1)
print(list1)
env=temp(list1)
env.add()
print(env.state)
print(list1)
list1 = list(list2)
print(list1)
