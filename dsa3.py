class Building:
    def _init_(self, id, name, location):
        self.id = id
        self.name = name
        self.location = location


class BSTNode:
    def _init_(self, building):
        self.building = building
        self.left = None
        self.right = None


class BST:
    def _init_(self):
        self.root = None

    def insert(self, building):
        self.root = self._insert(self.root, building)

    def _insert(self, node, building):
        if not node:
            return BSTNode(building)
        if building.name < node.building.name:
            node.left = self._insert(node.left, building)
        else:
            node.right = self._insert(node.right, building)
        return node

    def search(self, name):
        return self._search(self.root, name)

    def _search(self, node, name):
        if not node:
            return None
        if node.building.name == name:
            return node.building
        if name < node.building.name:
            return self._search(node.left, name)
        return self._search(node.right, name)

    def inorder(self, node):
        if node:
            self.inorder(node.left)
            print(node.building.name)
            self.inorder(node.right)

    def preorder(self, node):
        if node:
            print(node.building.name)
            self.preorder(node.left)
            self.preorder(node.right)

    def postorder(self, node):
        if node:
            self.postorder(node.left)
            self.postorder(node.right)
            print(node.building.name)


class AVLNode:
    def _init_(self, building):
        self.building = building
        self.left = None
        self.right = None
        self.height = 1


class AVLTree:
    def _height(self, node):
        return node.height if node else 0

    def _balance(self, node):
        return self._height(node.left) - self._height(node.right)

    def _right_rotate(self, y):
        x = y.left
        t2 = x.right
        x.right = y
        y.left = t2
        y.height = 1 + max(self._height(y.left), self._height(y.right))
        x.height = 1 + max(self._height(x.left), self._height(x.right))
        return x

    def _left_rotate(self, x):
        y = x.right
        t2 = y.left
        y.left = x
        x.right = t2
        x.height = 1 + max(self._height(x.left), self._height(x.right))
        y.height = 1 + max(self._height(y.left), self._height(y.right))
        return y

    def insert(self, root, building):
        if not root:
            return AVLNode(building)
        if building.name < root.building.name:
            root.left = self.insert(root.left, building)
        else:
            root.right = self.insert(root.right, building)
        root.height = 1 + max(self._height(root.left), self._height(root.right))
        balance = self._balance(root)
        if balance > 1 and building.name < root.left.building.name:
            return self._right_rotate(root)
        if balance < -1 and building.name > root.right.building.name:
            return self._left_rotate(root)
        if balance > 1 and building.name > root.left.building.name:
            root.left = self._left_rotate(root.left)
            return self._right_rotate(root)
        if balance < -1 and building.name < root.right.building.name:
            root.right = self._right_rotate(root.right)
            return self._left_rotate(root)
        return root


class Graph:
    def _init_(self, n):
        self.n = n
        self.adj_list = {i: [] for i in range(n)}
        self.adj_matrix = [[0] * n for _ in range(n)]

    def add_edge(self, u, v, w):
        self.adj_list[u].append((v, w))
        self.adj_list[v].append((u, w))
        self.adj_matrix[u][v] = w
        self.adj_matrix[v][u] = w

    def bfs(self, start):
        visited = [False] * self.n
        queue = [start]
        visited[start] = True
        order = []
        while queue:
            cur = queue.pop(0)
            order.append(cur)
            for nxt, _ in self.adj_list[cur]:
                if not visited[nxt]:
                    visited[nxt] = True
                    queue.append(nxt)
        print("BFS:", order)

    def dfs(self, start):
        visited = [False] * self.n
        stack = [start]
        order = []
        while stack:
            cur = stack.pop()
            if not visited[cur]:
                visited[cur] = True
                order.append(cur)
                for nxt, _ in reversed(self.adj_list[cur]):
                    if not visited[nxt]:
                        stack.append(nxt)
        print("DFS:", order)


def dijkstra(graph, start):
    dist = [float('inf')] * graph.n
    dist[start] = 0
    visited = [False] * graph.n
    for _ in range(graph.n):
        u = min((i for i in range(graph.n) if not visited[i]), key=lambda x: dist[x])
        visited[u] = True
        for v, w in graph.adj_list[u]:
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
    return dist


def kruskal(graph):
    edges = []
    for u in graph.adj_list:
        for v, w in graph.adj_list[u]:
            if u < v:
                edges.append((w, u, v))
    edges.sort()
    parent = list(range(graph.n))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    mst = []
    for w, u, v in edges:
        pu, pv = find(u), find(v)
        if pu != pv:
            mst.append((u, v, w))
            parent[pu] = pv
    return mst


class ExpNode:
    def _init_(self, val):
        self.val = val
        self.left = None
        self.right = None


def evaluate(root):
    if root.val.isdigit():
        return int(root.val)
    l = evaluate(root.left)
    r = evaluate(root.right)
    if root.val == '+': return l + r
    if root.val == '-': return l - r
    if root.val == '*': return l * r
    if root.val == '/': return l / r


if _name_ == "_main_":
    b1 = Building(1, "Library", "Block A")
    b2 = Building(2, "Cafeteria", "Block B")
    b3 = Building(3, "Lab", "Block C")

    print("\n--- BST ---")
    bst = BST()
    bst.insert(b1)
    bst.insert(b2)
    bst.insert(b3)
    bst.inorder(bst.root)

    print("\n--- AVL Tree Height Comparison ---")
    avl = AVLTree()
    root = None
    root = avl.insert(root, b1)
    root = avl.insert(root, b2)
    root = avl.insert(root, b3)
    print("AVL Height:", root.height)

    print("\n--- GRAPH ---")
    g = Graph(3)
    g.add_edge(0, 1, 5)
    g.add_edge(1, 2, 3)
    g.bfs(0)
    g.dfs(0)

    print("\nDijkstra:", dijkstra(g, 0))
    print("Kruskal MST:", kruskal(g))
