   ### Basic Data structures:
   ### Stack [LIFO]
   
       class Stack:
           def __init__(self):
               self.items = []
   
           def is_empty(self):
               return len(self.items) == 0
   
           def push(self, item):
               self.items.append(item)
   
           def pop(self):
               if not self.is_empty():
                   return self.items.pop()
               return "Stack is empty"
   
           def peek(self):
               if not self.is_empty():
                   return self.items[-1]
               return "Stack is empty"
   
           def size(self):
               return len(self.items)
   
       # Example usage
       stack = Stack()
       stack.push(1)
       stack.push(2)
       stack.push(3)
       print(stack.peek())  # Outputs 3
       print(stack.pop())   # Outputs 3
       print(stack.pop())   # Outputs 2
       print(stack.is_empty())  # Outputs False
       stack.pop()  # Removes 1
       print(stack.is_empty())  # Outputs True
   
   -   The `Stack` class uses a Python list (`self.items`) to store elements.
   -   `push` adds an element to the top of the stack.
   -   `pop` removes and returns the top element of the stack. If the stack is empty, it returns a message.
   -   `peek` returns the top element without removing it, or a message if the stack is empty.
   -   `is_empty` checks if the stack is empty.
   -   `size` returns the number of elements in the stack.
   
   ### Queue [FIFO]
   
       class Queue:
           def __init__(self):
               self.items = []
   
           def is_empty(self):
               return len(self.items) == 0
   
           def enqueue(self, item):
               self.items.append(item)
   
           def dequeue(self):
               if not self.is_empty():
                   return self.items.pop(0)
               return "Queue is empty"
   
           def peek(self):
               if not self.is_empty():
                   return self.items[0]
               return "Queue is empty"
   
           def size(self):
               return len(self.items)
   
       # Example usage
       queue = Queue()
       queue.enqueue('a')
       queue.enqueue('b')
       queue.enqueue('c')
       print(queue.peek())    # Outputs 'a'
       print(queue.dequeue()) # Outputs 'a'
       print(queue.dequeue()) # Outputs 'b'
       print(queue.is_empty()) # Outputs False
       queue.dequeue()  # Removes 'c'
       print(queue.is_empty()) # Outputs True
   
   -   The `Queue` class uses a Python list (`self.items`) to store the elements.
   -   `enqueue` adds an item to the end of the queue.
   -   `dequeue` removes and returns the item from the front of the queue. If the queue is empty, it returns a message.
   -   `peek` returns the item from the front of the queue without removing it, or a message if the queue is empty.
   -   `is_empty` checks if the queue is empty.
   -   `size` returns the number of items in the queue.
   
   While this implementation works, it's worth noting that dequeuing (removing an element from the front of the list) is not efficient in Python lists because it requires shifting all the other elements one position to the left. For a more efficient queue implementation, especially for larger datasets or frequent enqueue and dequeue operations, it is better to use `collections.deque` in Python which provides an optimized and faster way to handle a queue:
   
   	from collections import deque
   
   	class Queue:
   	    def __init__(self):
   	        self.items = deque()
   
   	    # Remaining methods are the same
   Using `deque` for a queue implementation provides O(1) time complexity for both enqueue and dequeue operations, making it a better choice for most queue-based applications.
   ### Linked lists
   	class Node:
   	    def __init__(self, data):
   	        self.data = data
   	        self.next = None
   
   	class LinkedList:
   	    def __init__(self):
   	        self.head = None
   
   	    def is_empty(self):
   	        return self.head is None
   
   	    def append(self, data):
   	        if self.head is None:
   	            self.head = Node(data)
   	        else:
   	            current = self.head
   	            while current.next:
   	                current = current.next
   	            current.next = Node(data)
   
   	    def display(self):
   	        elements = []
   	        current = self.head
   	        while current:
   	            elements.append(current.data)
   	            current = current.next
   	        return elements
   
   	    def size(self):
   	        count = 0
   	        current = self.head
   	        while current:
   	            count += 1
   	            current = current.next
   	        return count
   
   	# Example usage
   	ll = LinkedList()
   	ll.append(1)
   	ll.append(2)
   	ll.append(3)
   	print(ll.display())  # Outputs [1, 2, 3]
   	print(ll.size())     # Outputs 3
   -   `Node` class represents each element in the linked list, holding the data and a reference to the next node.
   -   `LinkedList` class manages the linked list, with methods to append new data, check if the list is empty, display all elements, and get the size of the list.
   -   `append` method adds a new node to the end of the list.
   -   `display` method returns a list of all elements in the linked list.
   -   `size` method returns the total number of nodes in the linked list.
   
   ### Tree
   
   	class TreeNode:
   	    def __init__(self, value):
   	        self.value = value
   	        self.left = None
   	        self.right = None
   
   	class BinaryTree:
   	    def __init__(self, root):
   	        self.root = TreeNode(root)
   
   	    def print_tree(self, traversal_type):
   	        if traversal_type == "preorder":
   	            return self.preorder_print(self.root, "")
   	        elif traversal_type == "inorder":
   	            return self.inorder_print(self.root, "")
   	        elif traversal_type == "postorder":
   	            return self.postorder_print(self.root, "")
   	        else:
   	            print("Traversal type " + str(traversal_type) + " is not supported.")
   	            return False
   
   	    def preorder_print(self, start, traversal):
   	        """Root -> Left -> Right"""
   	        if start:
   	            traversal += (str(start.value) + "-")
   	            traversal = self.preorder_print(start.left, traversal)
   	            traversal = self.preorder_print(start.right, traversal)
   	        return traversal
   
   	    def inorder_print(self, start, traversal):
   	        """Left -> Root -> Right"""
   	        if start:
   	            traversal = self.inorder_print(start.left, traversal)
   	            traversal += (str(start.value) + "-")
   	            traversal = self.inorder_print(start.right, traversal)
   	        return traversal
   
   	    def postorder_print(self, start, traversal):
   	        """Left -> Right -> Root"""
   	        if start:
   	            traversal = self.postorder_print(start.left, traversal)
   	            traversal = self.postorder_print(start.right, traversal)
   	            traversal += (str(start.value) + "-")
   	        return traversal
   
   	# Example Usage
   	#      1
   	#    /   \
   	#   2     3
   	#  / \   / \
   	# 4   5 6   7
   
   	tree = BinaryTree(1)
   	tree.root.left = TreeNode(2)
   	tree.root.right = TreeNode(3)
   	tree.root.left.left = TreeNode(4)
   	tree.root.left.right = TreeNode(5)
   	tree.root.right.left = TreeNode(6)
   	tree.root.right.right = TreeNode(7)
   
   	print(tree.print_tree("preorder"))  # Outputs 1-2-4-5-3-6-7-
   	print(tree.print_tree("inorder"))   # Outputs 4-2-5-1-6-3-7-
   	print(tree.print_tree("postorder")) # Outputs 4-5-2-6-7-3-1-
   In this implementation:
   
   -   `TreeNode` class represents each node in the tree, holding the value and references to the left and right children.
   -   `BinaryTree` class manages the binary tree, with a method to print the tree in different traversal orders: preorder, inorder, and postorder.
   
   ### Hash Table
   	class HashTable:
   	    def __init__(self, size=10):
   	        self.size = size
   	        self.table = [[] for _ in range(size)]
   
   	    def _hash_function(self, key):
   	        return hash(key) % self.size
   
   	    def set(self, key, value):
   	        hash_key = self._hash_function(key)
   	        key_exists = False
   	        bucket = self.table[hash_key]
   	        for i, kv in enumerate(bucket):
   	            k, v = kv
   	            if key == k:
   	                key_exists = True
   	                break
   	        if key_exists:
   	            bucket[i] = (key, value)
   	        else:
   	            bucket.append((key, value))
   
   	    def get(self, key):
   	        hash_key = self._hash_function(key)
   	        bucket = self.table[hash_key]
   	        for k, v in bucket:
   	            if key == k:
   	                return v
   	        return None
   
   	    def remove(self, key):
   	        hash_key = self._hash_function(key)
   	        bucket = self.table[hash_key]
   	        for i, kv in enumerate(bucket):
   	            k, v = kv
   	            if key == k:
   	                del bucket[i]
   
   	# Example usage
   	hash_table = HashTable()
   	hash_table.set("key1", "value1")
   	hash_table.set("key2", "value2")
   	print(hash_table.get("key1"))  # Outputs: value1
   	print(hash_table.get("key2"))  # Outputs: value2
   	hash_table.remove("key1")
   	print(hash_table.get("key1"))  # Outputs: None
   -   `HashTable` class creates a hash table with a specified size. Each slot in the hash table is initialized as an empty list (to handle collisions using chaining).
   -   `_hash_function` method is a simple hash function that uses Python's built-in `hash` function and modulates it with the size of the hash table to find the index for a given key.
   -   `set` method adds a key-value pair to the hash table. If the key already exists, it updates the value.
   -   `get` method retrieves a value by key. If the key is not found, it returns `None`.
   -   `remove` method deletes a key-value pair from the hash table.
   ### Search algorithms:
   ### Linear Search 
   	def linear_search(arr, target):
   	    for i in range(len(arr)):
   	        if arr[i] == target:
   	            return i  # Return the index where the target is found
   	    return -1  # Return -1 if the target is not found
   
   	# Example usage
   	my_list = [5, 3, 8, 6, 7, 2]
   	target = 6
   	result = linear_search(my_list, target)
   
   	if result != -1:
   	    print(f"Element found at index: {result}")
   	else:
   	    print("Element not found in the list")
   -   The function `linear_search` takes a list `arr` and a `target` value to find.
   -   It iterates through each element of the list using a `for` loop.
   -   If the element matches the `target`, the function returns the index of that element.
   -   If the `target` is not found by the end of the list, the function returns -1, indicating that the `target` is not present in the list.
   ### Binary Search 
   	def binary_search(arr, target):
   	    left, right = 0, len(arr) - 1
   
   	    while left <= right:
   	        mid = (left + right) // 2
   	        mid_value = arr[mid]
   
   	        if mid_value == target:
   	            return mid  # Target found, return its index
   	        elif mid_value < target:
   	            left = mid + 1  # Target is in the right half
   	        else:
   	            right = mid - 1  # Target is in the left half
   
   	    return -1  # Target is not in the list
   
   	# Example usage
   	my_sorted_list = [1, 3, 5, 7, 9, 11, 13, 15, 17]
   	target = 9
   	result = binary_search(my_sorted_list, target)
   
   	if result != -1:
   	    print(f"Element found at index: {result}")
   	else:
   	    print("Element not found in the list")
   -   The function `binary_search` takes a sorted list `arr` and a `target` value to find.
   -   It uses two pointers (`left` and `right`) to keep track of the portion of the list to be searched.
   -   It calculates the middle index `mid` and compares the element at `mid` with the `target`.
   -   If the `target` matches the `mid` element, it returns the index.
   -   If the `target` is less than the `mid` element, it repeats the search on the left half of the list. Otherwise, it searches the right half.
   -   If the `target` is not found, the function returns -1.
   ### Basic sorting algorithms:
   
   ### Selection sort
   	def selection_sort(arr):
   	    n = len(arr)
   
   	    for i in range(n):
   	        # Initially, assume the first element of the unsorted part as the minimum.
   	        min_index = i
   
   	        # Check the rest of the array for a smaller element.
   	        for j in range(i+1, n):
   	            if arr[j] < arr[min_index]:
   	                min_index = j
   
   	        # Swap the found minimum element with the first element.
   	        arr[i], arr[min_index] = arr[min_index], arr[i]
   
   	    return arr
   
   	# Example usage
   	my_list = [64, 25, 12, 22, 11]
   	sorted_list = selection_sort(my_list)
   	print("Sorted list:", sorted_list)
   
   -   The function `selection_sort` iterates through each element of the list.
   -   For each position `i` in the list, it finds the index `min_index` of the smallest element in the unsorted portion (from `i` to the end of the list).
   -   It then swaps the smallest found element with the element at position `i`.
   -   This process repeats for each position in the list, gradually moving all elements into their correct sorted position.
   
   ### Bubble sort
   	def bubble_sort(arr):
   	    n = len(arr)
   
   	    # Traverse through all elements in the array
   	    for i in range(n):
   	        # Last i elements are already in place, so the inner loop can avoid looking at them
   	        for j in range(0, n-i-1):
   	            # Traverse the array from 0 to n-i-1 and swap if the element found is greater than the next element
   	            if arr[j] > arr[j + 1]:
   	                arr[j], arr[j + 1] = arr[j + 1], arr[j]
   
   	    return arr
   
   	# Example usage
   	my_list = [64, 34, 25, 12, 22, 11, 90]
   	sorted_list = bubble_sort(my_list)
   	print("Sorted list:", sorted_list)
   
   -   The `bubble_sort` function iterates over each element in the array.
   -   In each iteration of the outer loop, the inner loop compares adjacent elements and swaps them if they are in the wrong order.
   -   With each pass through the array, the largest unsorted element "bubbles up" to its correct position at the end of the array.
   -   The process is repeated until no swaps are needed, which means the array is sorted.
   
   Bubble sort has a worst-case and average time complexity of `O(n^2)`, where n is the number of items being sorted.
   ### Insertion sort
   	def insertion_sort(arr):
   	    for i in range(1, len(arr)):
   	        key = arr[i]
   	        # Move elements of arr[0..i-1], that are greater than key,
   	        # to one position ahead of their current position
   	        j = i - 1
   	        while j >= 0 and key < arr[j]:
   	            arr[j + 1] = arr[j]
   	            j -= 1
   	        arr[j + 1] = key
   	    return arr
   
   	# Example usage
   	my_list = [12, 11, 13, 5, 6]
   	sorted_list = insertion_sort(my_list)
   	print("Sorted list:", sorted_list)
   -   The function `insertion_sort` iterates from the second element to the last element of the array.
   -   For each iteration, it stores the current element (`key`) and compares it with its preceding elements.
   -   Elements that are greater than the `key` are moved one position ahead in the array.
   -   Finally, the `key` is placed in its correct position in the sorted part of the array.
   
   Insertion sort is efficient for small data sets and is stable, meaning it does not change the relative order of elements with equal keys. Its time complexity is O(n^2) in the average and worst case, but it's O(n) in the best case (when the input is already sorted). Despite its simplicity, it's not suitable for sorting large unsorted lists compared to more advanced algorithms like mergesort or quicksort.
   
   ### Divide and conquer:
   ### Merge sort [Widely used across collection libraries]
   	def merge_sort(arr):
   	    if len(arr) > 1:
   	        # Finding the middle of the array
   	        mid = len(arr) // 2
   	        # Dividing the array elements into 2 halves
   	        left_half = arr[:mid]
   	        right_half = arr[mid:]
   
   	        # Sorting the first half
   	        merge_sort(left_half)
   	        # Sorting the second half
   	        merge_sort(right_half)
   
   	        # Merging the sorted halves
   	        i = j = k = 0
   
   	        # Copy data to temp arrays left_half[] and right_half[]
   	        while i < len(left_half) and j < len(right_half):
   	            if left_half[i] < right_half[j]:
   	                arr[k] = left_half[i]
   	                i += 1
   	            else:
   	                arr[k] = right_half[j]
   	                j += 1
   	            k += 1
   
   	        # Checking if any element was left in left_half
   	        while i < len(left_half):
   	            arr[k] = left_half[i]
   	            i += 1
   	            k += 1
   
   	        # Checking if any element was left in right_half
   	        while j < len(right_half):
   	            arr[k] = right_half[j]
   	            j += 1
   	            k += 1
   
   	    return arr
   
   	# Example usage
   	my_list = [38, 27, 43, 3, 9, 82, 10]
   	sorted_list = merge_sort(my_list)
   	print("Sorted list:", sorted_list)
   -   The `merge_sort` function recursively splits the list into halves until the sublists have only one element or are empty.
   -   These sublists are then merged together in a manner that results in a sorted list. This merging is done by comparing the smallest elements of each sublist and placing the smaller one into the new list, repeating this process until all elements are sorted and merged.
   
   Merge sort is notable for its efficiency—it has a time complexity of O(n log n), which makes it very efficient for large data sets. Moreover, it's a stable sort, preserving the order of equal elements, and works well for linked lists and random access data structures like arrays. However, unlike some other sorting algorithms like quicksort, merge sort requires O(n) extra space.
   
   ### Quick sort
   	def quick_sort(arr):
   	    if len(arr) <= 1:
   	        return arr
   	    else:
   	        pivot = arr[0]
   	        less_than_pivot = [x for x in arr[1:] if x <= pivot]
   	        greater_than_pivot = [x for x in arr[1:] if x > pivot]
   	        return quick_sort(less_than_pivot) + [pivot] + quick_sort(greater_than_pivot)
   
   	# Example usage
   	my_list = [99, 44, 6, 2, 1, 5, 63, 87, 283, 4, 0]
   	sorted_list = quick_sort(my_list)
   	print("Sorted list:", sorted_list)
   
   -   The function `quick_sort` checks if the list is empty or has one element (base case). If so, it returns the list as it is already sorted.
   -   It selects the first element as the pivot.
   -   Then it creates two sub-lists: one with elements less than or equal to the pivot and another with elements greater than the pivot.
   -   These sub-lists are then sorted recursively and combined with the pivot in between.
   
   QuickSort is generally very efficient with average and best-case time complexities of O(n log n), although its worst-case time complexity is O(n^2). However, in practice, its average performance is often better than other O(n log n) algorithms due to its low overhead and good cache performance.
   
   ### Introduction to graph theory: 
   **Key Concepts in Graph Theory:**
   
   1.  **Vertices (Nodes)**: The fundamental units or points, which can represent entities such as cities, computers, or intersections.
       
   2.  **Edges (Links)**: The connections between vertices, which can represent relationships or pathways like roads, friendships, or data transmission lines.
       
   3.  **Directed and Undirected Graphs**:
       
       -   **Undirected Graphs**: The edges have no direction. The connection between two vertices is bidirectional.
       -   **Directed Graphs (Digraphs)**: The edges have a direction, indicated by an arrow. The relationship is unidirectional, going from one vertex to another.
   4.  **Weighted and Unweighted Graphs**:
       
       -   **Unweighted Graphs**: Edges do not have any weight or cost associated with them.
       -   **Weighted Graphs**: Edges have a certain weight or cost, which could represent distance, time, or any other metric.
   5.  **Degree of a Vertex**: The number of edges connected to a vertex. In a directed graph, there are in-degrees (incoming edges) and out-degrees (outgoing edges).
       
   6.  **Path**: A sequence of edges that allows you to go from one vertex to another.
       
   7.  **Cycle**: A path that starts and ends at the same vertex, with all intermediate vertices being distinct.
       
   8.  **Connected and Disconnected Graphs**: In a connected graph, there is a path between every pair of vertices. In a disconnected graph, some vertices cannot be reached from others.
       
   9.  **Subgraph**: A graph whose sets of vertices and edges are subsets of another graph.
       
   10.  **Complete Graphs**: A graph in which every pair of vertices is connected by an edge.
   
   ### Basic graph algorithms: 
   ### Traversals
   > Depth-First Search (DFS)
   
   DFS explores as far as possible along each branch before backtracking. This can be implemented using `recursion` or a `stack`.
   	
   	def dfs(graph, start, visited=None):
   	    if visited is None:
   	        visited = set()
   	    visited.add(start)
   
   	    print(start, end=' ')
   
   	    for next in graph[start] - visited:
   	        dfs(graph, next, visited)
   	    return visited
   
   	# Example Usage
   	graph = {
   	    'A': set(['B', 'C']),
   	    'B': set(['A', 'D', 'E']),
   	    'C': set(['A', 'F']),
   	    'D': set(['B']),
   	    'E': set(['B', 'F']),
   	    'F': set(['C', 'E'])
   	}
   
   	dfs(graph, 'A')  # Starting from vertex 'A'
   >Breadth-First Search (BFS)
   
   BFS explores the neighbour vertices first, before moving to the next level neighbours. It uses a `queue` to keep track of the vertices to visit next.
   		
   	from collections import deque
   
   	def bfs(graph, start):
   	    visited, queue = set(), deque([start])
   	    visited.add(start)
   
   	    while queue:
   	        vertex = queue.popleft()
   	        print(vertex, end=' ')
   
   	        for neighbour in graph[vertex]:
   	            if neighbour not in visited:
   	                visited.add(neighbour)
   	                queue.append(neighbour)
   
   	# Example Usage
   	bfs(graph, 'A')  # Starting from vertex 'A'
   -   `graph` is represented as a dictionary with each vertex as a key and the set of its neighbors as values. This representation is known as an adjacency list.
   -   Both DFS and BFS print the order in which the nodes are visited.
   
   ### Shortest path
   The most famous algorithms for this task are Dijkstra's Algorithm (for graphs with non-negative edge weights) and the Bellman-Ford Algorithm (which also works with negative edge weights). 
   
   > Dijkstra's Algorithm
   
   Dijkstra's Algorithm finds the shortest path from a single source node to all other nodes in a graph with non-negative edge weights.
   
   	import heapq
   
   	def dijkstra(graph, start):
   	    # Initialize distances from start to all other nodes as infinity
   	    distances = {node: float('infinity') for node in graph}
   	    # Distance from start to itself is 0
   	    distances[start] = 0
   	    # Priority queue to hold vertices to be processed
   	    priority_queue = [(0, start)]
   
   	    while priority_queue:
   	        current_distance, current_node = heapq.heappop(priority_queue)
   
   	        # If a node's distance was updated after it was added to the queue, we can skip it
   	        if current_distance > distances[current_node]:
   	            continue
   
   	        for neighbor, weight in graph[current_node].items():
   	            distance = current_distance + weight
   
   	            # Only consider this new path if it's better than any path already found
   	            if distance < distances[neighbor]:
   	                distances[neighbor] = distance
   	                heapq.heappush(priority_queue, (distance, neighbor))
   	    
   	    return distances
   
   	# Example Usage
   	graph = {
   	    'A': {'B': 1, 'C': 4},
   	    'B': {'A': 1, 'C': 2, 'D': 5},
   	    'C': {'A': 4, 'B': 2, 'D': 1},
   	    'D': {'B': 5, 'C': 1}
   	}
   
   	print(dijkstra(graph, 'A'))  # Starting from vertex 'A'
   
   -   The graph is represented as a dictionary of dictionaries. The outer dictionary holds the nodes, and each inner dictionary holds the neighbors of that node and the weight of the edge to that neighbor.
   -   A priority queue (implemented using Python's `heapq` module) is used to select the node with the smallest tentative distance that hasn't been processed yet.
   -   The algorithm updates the shortest known distance to each node as it processes them.
   
   Dijkstra's Algorithm is very efficient for finding the shortest path in graphs with non-negative weights, with a time complexity of O(V + E log V), where V is the number of vertices and E is the number of edges. However, it's not suitable for graphs with negative edge weights. For such graphs, the Bellman-Ford Algorithm can be used.
   > Bellman-Ford Algorithm 
   
   The Bellman-Ford Algorithm is used for finding the shortest paths from a single source vertex to all other vertices in a weighted graph. It can handle graphs with negative weight edges, and it can also detect negative weight cycles. The algorithm is named after its developers, Richard Bellman and Lester Ford.
   
   	def bellman_ford(graph, source):
   	    # Step 1: Prepare the distance and predecessor for each node
   	    distance = {node: float('infinity') for node in graph}
   	    predecessor = {node: None for node in graph}
   
   	    # Initialize the source
   	    distance[source] = 0
   
   	    # Step 2: Relax the edges repeatedly
   	    for _ in range(len(graph) - 1):
   	        for node in graph:
   	            for neighbour in graph[node]:
   	                if distance[node] + graph[node][neighbour] < distance[neighbour]:
   	                    distance[neighbour] = distance[node] + graph[node][neighbour]
   	                    predecessor[neighbour] = node
   
   	    # Step 3: Check for negative weight cycles
   	    for node in graph:
   	        for neighbour in graph[node]:
   	            if distance[node] + graph[node][neighbour] < distance[neighbour]:
   	                print("Graph contains a negative weight cycle")
   	                return None
   
   	    return distance, predecessor
   
   	# Example usage
   	graph = {
   	    'A': {'B': 4, 'C': 2},
   	    'B': {'C': 3, 'D': 2, 'E': 3},
   	    'C': {'B': 1, 'D': 4, 'E': 5},
   	    'D': {},
   	    'E': {'D': -1}
   	}
   
   	distance, predecessor = bellman_ford(graph, 'A')
   	print("Distance:", distance)
   	print("Predecessor:", predecessor)
   -   The graph is represented as a dictionary of dictionaries. The outer dictionary holds the nodes, and each inner dictionary holds the neighbors of that node and the weight of the edge to that neighbor.
   -   It initializes the distance to all nodes as infinity and the distance to the source node as 0.
   -   The algorithm then relaxes each edge (i.e., checks if the known distance to a vertex can be improved by taking another route) repeatedly.
   -   After relaxing the edges, it checks for negative weight cycles. If it finds a cycle, it reports its presence, as the shortest path is undefined in such a scenario.
   
   The time complexity of the Bellman-Ford Algorithm is O(VE), where V is the number of vertices and E is the number of edges. This makes it less efficient than Dijkstra's algorithm for graphs without negative weight edges. However, its ability to handle negative weights makes it very versatile.

