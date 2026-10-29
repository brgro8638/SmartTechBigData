from heapq import heappop, heappush, heapify
from collections import defaultdict

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

# Encode a string using Huffman codes
def huffman_encode(input_string, code_dict):
    encoded_bits = []
    for char in input_string:
        encoded_bits.append(code_dict[char])
    return ''.join(encoded_bits)

# Decode a string using Huffman codes
def huffman_decode(encoded_bits, huffman_tree):
    decoded_string = ""
    current_node = huffman_tree
    for bit in encoded_bits:
        if bit == "0":
            current_node = current_node.left
        else:
            current_node = current_node.right

        if current_node.char is not None:
            decoded_string += current_node.char
            current_node = huffman_tree
    return decoded_string

# Compress string data using Huffman
def compress(data: str) -> tuple[str, Node]:
    # Build the frequency map
    freq_dict = defaultdict(int)
    for char in data:
        freq_dict[char] += 1

    # Build the Huffman tree
    huffman_tree = build_huffman_tree(freq_dict)

    # Generate Huffman codes for characters
    huffman_codes = {}
    generate_huffman_codes(huffman_tree, "", huffman_codes)

    # Encode the data using the Huffman codes
    encoded_bits = huffman_encode(data, huffman_codes)

    return encoded_bits, huffman_tree