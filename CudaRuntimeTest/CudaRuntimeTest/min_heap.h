#pragma once

/**
    Code for https://journaldev.com article
    Purpose: A Min Heap Implementation, representing the Binary Tree as an Array
    @author: Vijay Ramachandran
    @date: 27-02-2020
*/

#include <stdio.h>
#include <stdlib.h>

typedef struct MinHeap MinHeap;
struct MinHeap {
    int* arr;
    // Current Size of the Heap
    int size;
    // Maximum capacity of the heap
    int capacity;
};

inline int parent(int i) {
    // Get the index of the parent
    return (i - 1) / 2;
}

inline int left_child(int i) {
    return (2 * i + 1);
}

inline int right_child(int i) {
    return (2 * i + 2);
}

inline int get_min(MinHeap* heap) {
    // Return the root node element,
    // since that's the minimum
    return heap->arr[0];
}

inline MinHeap* init_minheap(int capacity) {
    MinHeap* minheap = (MinHeap*)calloc(1, sizeof(MinHeap));
    minheap->arr = (int*)calloc(capacity, sizeof(int));
    minheap->capacity = capacity;
    minheap->size = 0;
    return minheap;
}

inline MinHeap* insert_minheap(MinHeap* heap, int element) {
    // Inserts an element to the min heap
    // We first add it to the bottom (last level)
    // of the tree, and keep swapping with it's parent
    // if it is lesser than it. We keep doing that until
    // we reach the root node. So, we will have inserted the
    // element in it's proper position to preserve the min heap property
    if (heap->size == heap->capacity) {
        fprintf(stderr, "Cannot insert %d. Heap is already full!\n", element);
        return heap;
    }
    // We can add it. Increase the size and add it to the end
    heap->size++;
    heap->arr[heap->size - 1] = element;

    // Keep swapping until we reach the root
    int curr = heap->size - 1;
    // As long as you aren't in the root node, and while the 
    // parent of the last element is greater than it
    while (curr > 0 && heap->arr[parent(curr)] > heap->arr[curr]) {
        // Swap
        int temp = heap->arr[parent(curr)];
        heap->arr[parent(curr)] = heap->arr[curr];
        heap->arr[curr] = temp;
        // Update the current index of element
        curr = parent(curr);
    }
    return heap;
}

inline MinHeap* heapify(MinHeap* heap, int index) {
    // Rearranges the heap as to maintain
    // the min-heap property
    if (heap->size <= 1)
        return heap;

    int left = left_child(index);
    int right = right_child(index);

    // Variable to get the smallest element of the subtree
    // of an element an index
    int smallest = index;

    // If the left child is smaller than this element, it is
    // the smallest
    if (left < heap->size && heap->arr[left] < heap->arr[index])
        smallest = left;

    // Similarly for the right, but we are updating the smallest element
    // so that it will definitely give the least element of the subtree
    if (right < heap->size && heap->arr[right] < heap->arr[smallest])
        smallest = right;

    // Now if the current element is not the smallest,
    // swap with the current element. The min heap property
    // is now satisfied for this subtree. We now need to
    // recursively keep doing this until we reach the root node,
    // the point at which there will be no change!
    if (smallest != index)
    {
        int temp = heap->arr[index];
        heap->arr[index] = heap->arr[smallest];
        heap->arr[smallest] = temp;
        heap = heapify(heap, smallest);
    }

    return heap;
}

inline MinHeap* delete_minimum(MinHeap* heap) {
    // Deletes the minimum element, at the root
    if (!heap || heap->size == 0)
        return heap;

    int size = heap->size;
    int last_element = heap->arr[size - 1];

    // Update root value with the last element
    heap->arr[0] = last_element;

    // Now remove the last element, by decreasing the size
    heap->size--;
    size--;

    // We need to call heapify(), to maintain the min-heap
    // property
    heap = heapify(heap, 0);
    return heap;
}

inline MinHeap* delete_element(MinHeap* heap, int index) {
    // Deletes an element, indexed by index
    // Ensure that it's lesser than the current root
    heap->arr[index] = get_min(heap) - 1;

    // Now keep swapping, until we update the tree
    int curr = index;
    while (curr > 0 && heap->arr[parent(curr)] > heap->arr[curr]) {
        int temp = heap->arr[parent(curr)];
        heap->arr[parent(curr)] = heap->arr[curr];
        heap->arr[curr] = temp;
        curr = parent(curr);
    }

    // Now simply delete the minimum element
    heap = delete_minimum(heap);
    return heap;
}

inline void print_heap(MinHeap* heap) {
    // Simply print the array. This is an
    // inorder traversal of the tree
    printf("Min Heap:\n");
    for (int i = 0; i < heap->size; i++) {
        printf("%d -> ", heap->arr[i]);
    }
    printf("\n");
}

inline void free_minheap(MinHeap* heap) {
    if (!heap)
        return;
    free(heap->arr);
    free(heap);
}

//int main() {
//    // Capacity of 10 elements
//    MinHeap* heap = init_minheap(10);
//
//    insert_minheap(heap, 40);
//    insert_minheap(heap, 50);
//    insert_minheap(heap, 5);
//    print_heap(heap);
//
//    // Delete the heap->arr[1] (50)
//    delete_element(heap, 1);
//
//    print_heap(heap);
//    free_minheap(heap);
//    return 0;
//}