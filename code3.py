'''from collections import defaultdict
def decoder(s): 
    freq = {}
    for char in s:  
        if char in freq:
            freq[char] += 1 
        else:
            freq[char] = 1 
    sorted_freq = sorted(freq.items())   
    return sorted_freq
def main():
    s = input() 
    ret = decoder(s)  
    for char,count in ret:
        print(f"{char}:{count}",end=" ")
        
main()'''
'''import multiprocessing
import ctypes

def compare_and_swap(shared_arr, i, n):
    if i + 1 < n and shared_arr[i] > shared_arr[i + 1]:
        shared_arr[i], shared_arr[i + 1] = shared_arr[i + 1], shared_arr[i]

def parallel_bubble_sort(arr):
    n = len(arr)
    # Create shared memory array
    shared_arr = multiprocessing.Array(ctypes.c_int, arr)

    for i in range(n):
        processes = []
        for j in range(i % 2, n - 1, 2):
            p = multiprocessing.Process(target=compare_and_swap, args=(shared_arr, j, n))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    # Copy results back
    arr[:] = shared_arr[:]

if __name__ == "__main__":
    arr = [29, 10, 14, 37, 13, 5, 42, 1]
    print("Original array:", arr)
    parallel_bubble_sort(arr)
    print("Sorted array:  ", arr)'''


#Implementation of parallel BFS algorithm using multiprocessing in Python 

'''from collections import deque
import multiprocessing
from concurrent.futures import ThreadPoolExecutor 

def bfs_parallel(graph, start_node):
    visited = set()
    visited.add(start_node)
    queue = deque([start_node])
    ret = [] 
    
    
    while queue:
        lvl_size = len(queue)
        cur_lvl_nodes = [] 
        for _ in range(lvl_size):
            node = queue.popleft()
            ret.append(node)
            cur_lvl_nodes.extend(graph[node])
        
        nxt_lvl_nodes = [node for node in cur_lvl_nodes if node not in visited]
        visited.update(nxt_lvl_nodes)
        
        with ThreadPoolExecutor() as executor:
            future_results = [executor.submit(queue.append,node) for node in nxt_lvl_nodes]
            for future in future_results:
                future.result() 
                
    return ret 

def main():
    graph = {
        0: [1, 2],
        1: [0, 3, 4],
        2: [0, 5],
        3: [1],
        4: [1],
        5: [2]
    }
    
    # Perform Parallel BFS starting from node 0
    result = bfs_parallel(graph, 0)
    print("BFS Traversal Order:", result) 
    
main()'''


#Implementation of parallel DFS algorithm using multiprocessing in Python 

'''from collections import deque
import multiprocessing 
from concurrent.futures import ThreadPoolExecutor 

def bfs_parallel(grph,start_node):
    visited = set() 
    visited.add(start_node) 
    queue = deque([start_node]) 
    ret = [] 
    while queue:
        lvl_size = len(queue) 
        cur_lvl_nodes = [] 
        for _ in range(lvl_size):
            node = queue.popleft()
            ret.append(node)
            cur_lvl_nodes.extend(grph[node]) 
        nxt_lvl_nodes  = [node for node in cur_lvl_nodes if node not in visited] 
        visited.update(nxt_lvl_nodes) 
        with ThreadPoolExecutor() as executor:
            future_results = [executor.submit(queue.append,node)for node in nxt_lvl_nodes] 
            for future in future_results:
                future.result() 
    return ret
def get_graph_from_input():
    # Get the number of nodes in the graph
    num_nodes = int(input("Enter the number of nodes in the graph: "))
    
    # Initialize the graph as an empty dictionary
    graph = {i: [] for i in range(num_nodes)}
    
    # Get the edges as input
    num_edges = int(input("Enter the number of edges: "))
    
    print("Enter each edge as a pair of nodes (e.g., '0 1' for an edge between nodes 0 and 1):")
    for _ in range(num_edges):
        u, v = map(int, input().split())  # Read the edge
        graph[u].append(v)
        graph[v].append(u)  # Since the graph is undirected
    
    return graph

def main():
    # Get the graph from user input
    graph = get_graph_from_input()
    
    # Ask for the starting node for BFS
    start_node = int(input("Enter the start node for BFS: "))
    
    # Perform Parallel BFS starting from the start node
    result = bfs_parallel(graph, start_node)
    print("BFS Traversal Order:", result)

# Run the main function
main()''' 

'''from concurrent.futures import ThreadPoolExecutor
import time
import random
import threading

# Lock to ensure thread safety during swaps
swap_lock = threading.Lock()

def bubble_sort_sequential(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def bubble_sort_parallel(arr):
    n = len(arr)
    for i in range(n):
        with ThreadPoolExecutor() as executor:
            future_results = []
            for j in range(0, n - i - 1):
                # Submit each comparison task to the executor
                future_results.append(executor.submit(swap_elements, arr, j, j + 1))

            # Wait for all futures to complete
            for future in future_results:
                future.result()

def swap_elements(arr, i, j):
    """Helper function to swap elements in the array with a lock."""
    with swap_lock:  # Ensure that only one thread modifies the array at a time
        if arr[i] > arr[j]:
            arr[i], arr[j] = arr[j], arr[i]

def measure_time(sort_function, arr):
    start = time.time()
    sort_function(arr)
    return time.time() - start

def main():
    arr_size = 1000
    arr = [random.randint(0, 1000) for _ in range(arr_size)]
    
    # Sequential Sort
    arr_seq = arr.copy()
    seq_time = measure_time(bubble_sort_sequential, arr_seq)
    print(f"Sequential Bubble Sort Time: {seq_time:.6f} seconds")
    print("Sorted Array (Sequential):")
    print(arr_seq[:100])  # Print first 100 elements to avoid large output
    
    # Parallel Sort
    arr_par = arr.copy()
    par_time = measure_time(bubble_sort_parallel, arr_par)
    print(f"Parallel Bubble Sort Time: {par_time:.6f} seconds")
    print("Sorted Array (Parallel):")
    print(arr_par[:100])  # Print first 100 elements to avoid large output

if __name__ == "__main__":
    main()''' 
    
    
import multiprocessing
from concurrent.futures import ThreadPoolExecutor 
import random,time 

'''def min_parallel(arr):
    def min_reduction(arr_chunk):  
        return min(arr_chunk) 
    chunk_size = len(arr) // 4  # Split the array into 4 chunks (adjustable)
    chunks = [arr[i:i + chunk_size] for i in range(0, len(arr), chunk_size)]

    with ThreadPoolExecutor() as executor:
        # Map each chunk to the min_reduction function
        partial_results = list(executor.map(min_reduction, chunks))

    # Reduce (combine) the results to get the final min value
    return min(partial_results) 

        
def max_parallel():
    pass

def sum_parallel():
    pass
def avg_parallel():
    pass

def measur_time(func,arr): 
    start = time.time()
    ret = func(arr)
    end = time.time() 
    return ret, end - start
    
def main():
    arr_size = 100000
    arr = [random.randint(1, 1000) for _ in range(arr_size)]

    # Min operation
    min_result, min_time = measure_time(min_parallel, arr)
    print(f"Parallel Min: {min_result}, Time: {min_time:.6f} seconds")

    # Max operation
    max_result, max_time = measure_time(max_parallel, arr)
    print(f"Parallel Max: {max_result}, Time: {max_time:.6f} seconds")

    # Sum operation
    sum_result, sum_time = measure_time(sum_parallel, arr)
    print(f"Parallel Sum: {sum_result}, Time: {sum_time:.6f} seconds")

    # Average operation
    avg_result, avg_time = measure_time(avg_parallel, arr)
    print(f"Parallel Average: {avg_result:.2f}, Time: {avg_time:.6f} seconds")
main()'''

import random
import time
from concurrent.futures import ThreadPoolExecutor

'''def parallel_reduction(arr):
    # Split the array into 4 chunks (or you can adjust n as needed)
    n_chunks = 4
    chunks = chunkify(arr, n_chunks)

    # Create a ThreadPoolExecutor to apply the min, max, sum reduction in parallel
    with ThreadPoolExecutor() as executor:
        # Perform reduction on each chunk in parallel
        min_val = min(executor.map(min, chunks))
        max_val = max(executor.map(max, chunks))
        sum_val = sum(executor.map(sum, chunks))

    # Calculate the average based on the sum and length of the original array
    avg_val = sum_val / len(arr)

    return min_val, max_val, sum_val, avg_val

def chunkify(lst, n=4):
    # This will divide the list into `n` approximately equal chunks
    avg = len(lst) // n
    chunks = [lst[i:i + avg] for i in range(0, len(lst), avg)]
    if len(chunks) > n:
        # In case the chunks aren't evenly divided, merge last chunks
        chunks[-2].extend(chunks[-1])
        chunks = chunks[:-1]
    return chunks

def measure_time(func, arr):
    start = time.time()
    result = func(arr)
    end = time.time()
    return result, end - start

# Main function to demonstrate parallel reduction
def main():
    arr_size = 1000
    arr = [random.randint(1, 1000) for _ in range(arr_size)]
    
    # Perform parallel reduction
    result, time_taken = measure_time(parallel_reduction, arr)
    min_val, max_val, sum_val, avg_val = result
    print(f"Parallel Reduction Time: {time_taken:.6f} seconds")
    print(f"Min: {min_val}, Max: {max_val}, Sum: {sum_val}, Average: {avg_val:.2f}")

if __name__ == "__main__":
    main()'''

#Parallel reduction of an array using ThreadPoolExecutor in Python
'''def parallel_reduction(arr):
    n_chunks = 4 
    chunks = chunkify(arr,n_chunks) 
    with ThreadPoolExecutor() as executor:
        min_val = min(executor.map(min,chunks)) 
        max_val = max(executor.map(max,chunks)) 
        sum_val = sum(executor.map(sum,chunks)) 
    
    avg_val = sum_val/len(arr) 
    return min_val,max_val,sum_val,avg_val 

def chunkify(lst,n=4):
    avg = len(lst)//n
    chunks = [lst[i:i + avg] for i in range(0,len(lst),avg)] 
    if len(chunks) > n:
        chunks[-2].extend(chunks[-1]) 
        chunks = chunks[:-1] 
    return chunks  

def measure_time(func,arr):
    start = time.time() 
    ret = func(arr)
    end = time.time() 
    return ret, end-start 

def main():
    arr_size = 1000
    arr = [random.randint(1,1000) for _ in range(arr_size)]
    ret,time_taken = measure_time(parallel_reduction,arr)
    min_val,max_val,sum_val,avg_val = ret 
    print(f"parrallel Reduction Time:{time_taken:.6f} seconds") 
    print(f"MIN: {min_val}, MAX: {max_val}, SUM: {sum_val}, AVERAGE: {avg_val:.2f}") 
main()''' 


#Implementation of sequential and parallel merge sort
import random
import time
from concurrent.futures import ThreadPoolExecutor

# Sequential Merge Sort
def merge_sort_sequential(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]

        # Recursively sort each half
        merge_sort_sequential(left)
        merge_sort_sequential(right)

        # Merge the sorted halves
        merge(arr, left, right)

def merge(arr, left, right):
    i = j = k = 0
    # Merge the left and right subarrays into the original array
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1

    # If there are remaining elements in left, append them
    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1

    # If there are remaining elements in right, append them
    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1

# Parallel Merge Sort
def parallel_merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    # Use ThreadPoolExecutor to parallelize the sorting of each half
    with ThreadPoolExecutor() as executor:
        left_future = executor.submit(parallel_merge_sort, left)
        right_future = executor.submit(parallel_merge_sort, right)
        left_sorted = left_future.result()
        right_sorted = right_future.result()

    # Merge the sorted halves
    return merge_sorted(left_sorted, right_sorted)

def merge_sorted(left, right):
    merged = []
    i = j = 0
    # Merge the left and right subarrays into the merged array
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    # If there are remaining elements in left, append them
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged

# Measure time taken by the sorting algorithm
def measure_time(func, arr):
    start = time.time()
    result = func(arr)
    end = time.time()
    return result, end - start

# Main function to demonstrate both algorithms
def main():
    arr_size = 1000
    arr = [random.randint(1, 1000) for _ in range(arr_size)]

    # Sequential Merge Sort
    arr_seq = arr.copy()
    seq_result, seq_time = measure_time(merge_sort_sequential, arr_seq)
    print(f"Sequential Merge Sort Time: {seq_time:.6f} seconds")
    print(arr_seq[:100])  # Print first 100 elements to avoid large output

    # Parallel Merge Sort
    arr_par = arr.copy()
    par_result, par_time = measure_time(parallel_merge_sort, arr_par)
    print(f"Parallel Merge Sort Time: {par_time:.6f} seconds")
    print(arr_par[:100])  # Print first 100 elements to avoid large output

    # Verify both results are the same
    assert seq_result == par_result, "The results of the sequential and parallel mergesort don't match!"

if __name__ == "__main__":
    main()
