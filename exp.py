def recBananaCount(A,B,C):
    if B <= A:
        return 0
    if B <= C:
        return B-A
    if A == 0:
        return B
    if dp[A][B] != -1:
        return dp[A][B]
    
    maxCount = -2**32
    tripCount = ((2*B) // C)-1 if (B%C==0) else ((2*B) // C)+1
    
    for i in range(1,A+1):
        curCount = recBananaCount(A-i, B-tripCount*i, C)
        if curCount > maxCount:
            maxCount = curCount
            dp[A][B] = maxCount
    return maxCount

A = int(input("Enter the total distance: "))
B = int(input("Enter the total number of bananas: "))
C = int(input("Enter the number of bananas the camel can carry at a time: "))

dp = [[-1 for i in range(B+1)] for j in range(C+1)]
print(recBananaCount(A,B,C))


import random
class Thermostat:
    def __init__(self, target_temperature):
        self.target_temperature = target_temperature
        self.current_temperature = 0
        self.heating = False
        self.cooling = False

    def update_temperature(self, new_temperature):
        self.current_temperature = new_temperature
        if self.current_temperature < self.target_temperature:
            self.heating = True
            self.cooling = False
            print("Heating turned on")
        elif self.current_temperature > self.target_temperature:
            self.heating = False
            self.cooling = True
            print("Cooling turned on")
        else:
            self.heating = False
            self.cooling = False
            print("Temperature at target level, no action needed")

inti=70

thermostat = Thermostat(inti)  


curr = random.randrange(30,80)
print(curr)
thermostat.update_temperature(curr) 


def CheckLatinSquare(mat):    
    N = len(mat)
    rows = []
    for i in range(N):
        rows.append(set([]))
    cols = []
    for i in range(N):
        cols.append(set([]))
    invalid = 0
  
    for i in range(N):
        for j in range(N):
            rows[i].add(mat[i][j])
            cols[j].add(mat[i][j])
  
            if (mat[i][j] > N or mat[i][j] <= 0) :
                invalid += 1
    
    numrows = 0
    numcols = 0
  
    for i in range(N):
        if (len(rows[i]) != N) :
            numrows+=1
        if (len(cols[i]) != N) :
            numcols+=1
  
    if (numcols == 0 or numrows == 0 and invalid == 0) :
        print("YES")
    else:
        print("NO")
    return
 
Matrix = [[ 1, 2, 3, 4 ],
          [ 2, 1, 4, 3 ],
          [ 3, 4, 1, 2 ],
          [ 4, 3, 2, 1 ]]
CheckLatinSquare(Matrix)


from collections import deque

def bfs(graph,start):
    visited=set()
    queue=deque([start])
    
    while queue:
        node=queue.popleft()
        if node not in visited:
            visited.add(node)
            print(node, end=' ')
            queue.extend(graph[node])

def dfs(graph,start):
    visited=set()
    stack=[start]
    
    while stack:
        node=stack.pop()
        if node not in visited:
            visited.add(node)
            print(node, end=' ')
            stack.extend(graph[node])

graph = {
    'A' : ['B', 'C'],
    'B' : ['A', 'D', 'E'],
    'C' : ['A', 'F', 'G'],
    'D' : ['B'],
    'E' : ['B', 'H'],
    'F' : ['C'],
    'G' : ['C'],
    'H' : ['E']
}

print("bfs:")
bfs(graph,'A')
print()
print("dfs:")
dfs(graph,'A')


from queue import PriorityQueue
v = 14
graph = [[] for i in range(v)]
def best_first(source, destination, graph):
    visited = [False] * 14
    pq = PriorityQueue()
    pq.put((0, source))
    visited[source] = True
    while pq.empty() == False:
        node = pq.get()[1]
        print(node, end = " ")
        if node == destination:
            break
        for v,c in graph[node]:
            if visited[v] == False:
                visited[v] = True
                pq.put((c,v))
    print()
def addEdge(x, y, cost):
    graph[x].append((y, cost))
    graph[y].append((x, cost))
addEdge(0, 1, 3)
addEdge(0, 2, 6)
addEdge(0, 3, 5)
addEdge(1, 4, 9)
addEdge(1, 5, 8)
addEdge(2, 6, 12)
addEdge(2, 7, 14)
addEdge(3, 8, 7)
addEdge(8, 9, 5)
addEdge(8, 10, 6)
addEdge(9, 11, 1)
addEdge(9, 12, 12)
addEdge(9, 13, 2)
source = 0
target = 9
best_first(source, target, graph)


import heapq
def astar(graph, start, goal):
    frontier = [(0, start)] 
    came_from = {}
    cost_so_far = {start: 0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for next_node in graph[current]:
            new_cost = cost_so_far[current] + graph[current][next_node]
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(next_node, goal)
                heapq.heappush(frontier, (priority, next_node))
                came_from[next_node] = current
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path
def heuristic(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])
if __name__ == "__main__":
    graph = {
        (0, 0): {(0, 1): 1, (1, 0): 1},
        (0, 1): {(0, 0): 1, (0, 2): 1},
        (0, 2): {(0, 1): 1, (1, 2): 1},
        (1, 0): {(0, 0): 1, (1, 1): 1},
        (1, 1): {(1, 0): 1, (1, 2): 1},
        (1, 2): {(0, 2): 1, (1, 1): 1}
    }
    start = (0, 0)
    goal = (1, 2)
    path = astar(graph, start, goal)
    print("Path found by A* Search:")
    print(path)


import random
def play_monty_hall (choice):
    prizes=['goat', 'car', 'goat']
    random.shuffle(prizes)
    while True:
        opening_door = random.randrange (len(prizes))
        if prizes [opening_door] != 'car' and choice - 1 != opening_door:
            break
        opening_door += 1
    print("We are opening door number: ", opening_door + 1)
    options = [1, 2, 3]
    options.remove(choice)
    options.remove(opening_door + 1)
    switching_door = options[0]
    print("Now, do you want to switch to door number: ",switching_door ," ?(yes/no)")
    answer = input()
    if answer.lower() == 'yes':
        result = switching_door - 1
    else:
        result = choice - 1
    print("And your prize is..." , prizes [result].upper())
choice = int(input("Which door do you want to choose? (1, 2, 3): "))
play_monty_hall(choice)


def unify(statement1, statement2):
    words1 = statement1.split()
    words2 = statement2.split()
    substitution = {}
    for word1, word2 in zip(words1, words2):
        if word2.isalpha() and word2[0].isupper():
            substitution[word2] = word1
        elif word1 != word2:
            return None
    return substitution
statement1 = "Moksha and Vineeta are sisters"
statement2 = "X and Y are sisters"
result = unify(statement1, statement2)
if result:
    print("The unification is successful. Substitution = ", result)
else:
    print("Unification failed")


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

data.head(5)

X = data[['Age','Annual Income (k$)','Spending Score (1-100)' ]]
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
kmeans=KMeans(n_clusters=4,init='k-means++',random_state=42)
kmeans.fit(X)

plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=kmeans.labels_, cmap='viridis')
##plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300)
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()





import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

text="The quick brown fox jumps over the lazy dog."

tokens=word_tokenize(text)
print(tokens)

stop_words = set(stopwords.words('english'))
toks = [token for token in tokens if token.lower() not in stop_words]
print(toks)

ltoks=[WordNetLemmatizer().lemmatize(token) for token in toks]
print(ltoks)

stoks = [PorterStemmer().stem(token) for token in toks]
print(stoks)

ptags=pos_tag(toks)
print(ptags)



from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense

dataset = load_breast_cancer()

print(dataset.DESCR)

features = dataset.data
target = dataset.target

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2)

model = Sequential()
model.add(Dense(32, input_dim = 30, activation = 'relu')) ## hidden layer 1
model.add(Dense(64, activation = 'relu')) ## hidden layer 2
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 10)

scores = model.evaluate(X_test, y_test)
print(scores)

predictions = model.predict(X_test)
label = []
for pred in predictions:
  if pred>=0.5:
    print("Malignant")
  else:
    print("Benign")


