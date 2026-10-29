class Node {
    constructor(byteVal, freq) {
        this.byte = byteVal; // number or null
        this.freq = freq;
        this.left = null;
        this.right = null;
    }
}

class Huffman {
    static buildHuffmanTree(freqDict) {
        // Priority queue sorted by frequency ascending
        // Node elements are kept sorted using a simple inserted-sort array
        const pq = [];

        for (const [bStr, freq] of freqDict.entries()) {
            const b = Number(bStr);
            pq.push(new Node(b, freq));
        }

        pq.sort((a, b) => a.freq - b.freq);

        // Edge case: data contains only one unique byte
        if (pq.length === 1) {
            const singleNode = pq.shift();
            const parentNode = new Node(null, singleNode.freq);
            parentNode.left = singleNode;
            return parentNode;
        }

        while (pq.length > 1) {
            const left = pq.shift();
            const right = pq.shift();

            const combined = new Node(null, left.freq + right.freq);
            combined.left = left;
            combined.right = right;

            // Insert combined node back maintaining sorted order
            let inserted = false;
            for (let i = 0; i < pq.length; i++) {
                if (combined.freq < pq[i].freq) {
                    pq.splice(i, 0, combined);
                    inserted = true;
                    break;
                }
            }
            if (!inserted) pq.push(combined);
        }

        return pq.length > 0 ? pq.shift() : null;
    }

    static generateHuffmanCodes(root) {
        const codeDict = new Map();
        
        function traverse(node, code) {
            if (node === null) return;

            if (node.byte !== null && node.byte !== undefined) {
                codeDict.set(node.byte, code);
                return;
            }

            traverse(node.left, code + "0");
            traverse(node.right, code + "1");
        }

        traverse(root, "");
        return codeDict;
    }

    static compress(inputData, codeDict) {
        const outputBytes = [];
        let currentByte = 0;
        let bitCount = 0;
        let bitLength = 0;

        for (const b of inputData) {
            const code = codeDict.get(b);
            for (let i = 0; i < code.length; i++) {
                const bitChar = code[i];
                currentByte <<= 1;
                if (bitChar === '1') {
                    currentByte |= 1;
                }
                bitCount++;
                bitLength++;

                if (bitCount === 8) {
                    outputBytes.push(currentByte & 0xFF);
                    currentByte = 0;
                    bitCount = 0;
                }
            }
        }

        // Handle remaining leftover bits
        if (bitCount > 0) {
            currentByte <<= (8 - bitCount);
            outputBytes.push(currentByte & 0xFF);
        }

        return {
            encodedBytes: Uint8Array.from(outputBytes),
            bitLength: bitLength
        };
    }

    static decode(compressedBits, bitLength, huffmanTree) {
        if (!huffmanTree || bitLength === 0) return new Uint8Array(0);

        const decodedBytes = [];
        let currentNode = huffmanTree;
        let processedBits = 0;

        for (const b of compressedBits) {
            for (let i = 7; i >= 0; i--) {
                if (processedBits >= bitLength) break;

                const bit = (b >> i) & 1;
                currentNode = (bit === 0) ? currentNode?.left : currentNode?.right;

                if (currentNode && currentNode.byte !== null && currentNode.byte !== undefined) {
                    decodedBytes.push(currentNode.byte);
                    currentNode = huffmanTree;
                }

                processedBits++;
            }
        }

        return Uint8Array.from(decodedBytes);
    }

    // Handles the complete process from input to output
    static encode(data) {
        // 1. Build frequency map
        const freqDict = new Map();
        for (const b of data) {
            freqDict.set(b, (freqDict.get(b) || 0) + 1);
        }

        // 2. Build Huffman tree
        const huffmanTree = Huffman.buildHuffmanTree(freqDict);

        // 3. Generate Huffman codes
        const huffmanCodes = Huffman.generateHuffmanCodes(huffmanTree);

        // 4. Compress data into packed Uint8Array
        const { encodedBytes, bitLength } = Huffman.compress(data, huffmanCodes);

        return {
            encoded: encodedBytes,
            bitLength: bitLength,
            tree: huffmanTree
        };
    }
}