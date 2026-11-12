from heapq import heappop, heappush, heapify

# Define a Node class for the Huffman tree
class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

# Build the Huffman tree
def build_huffman_tree(freq_dict):
    heap = [Node(char, freq) for char, freq in freq_dict.items()]
    heapify(heap)

    while len(heap) > 1:
        left = heappop(heap)
        right = heappop(heap)

        combined = Node(None, left.freq + right.freq)
        combined.left = left
        combined.right = right

        heappush(heap, combined)

    return heap[0]

# Generate Huffman codes
def generate_huffman_codes(node, code="", code_dict={}):
    if node is None:
        return

    if node.char is not None:
        code_dict[node.char] = code
        return

    generate_huffman_codes(node.left, code + "0", code_dict)
    generate_huffman_codes(node.right, code + "1", code_dict)