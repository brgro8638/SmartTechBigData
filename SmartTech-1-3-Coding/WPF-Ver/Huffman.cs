using System.IO;

namespace ST_1_3_Coding
{
    public class Huffman
    {
        public class Node(byte? b, int freq)
        {
            public byte? Byte { get; set; } = b;
            public int Freq { get; set; } = freq;
            public Node? Left { get; set; } = null;
            public Node? Right { get; set; } = null;
        }

        public static Node? BuildHuffmanTree(Dictionary<byte, int> freqDict)
        {
            PriorityQueue<Node, int> pq = new();

            foreach (var (b, freq) in freqDict)
            {
                pq.Enqueue(new Node(b, freq), freq);
            }

            // edge case hvor data består av bare et unikt element (må forhindre problem ved pq.Count > 1 senere, som fører til null return)
            if (pq.Count == 1)
            {
                var singleNode = pq.Dequeue();
                var parentNode = new Node(null, singleNode.Freq)
                {
                    Left = singleNode
                };
                return parentNode;
            }

            while (pq.Count > 1)
            {
                var left = pq.Dequeue();
                var right = pq.Dequeue();

                var combined = new Node(null, left.Freq + right.Freq)
                {
                    Left = left,
                    Right = right
                };

                pq.Enqueue(combined, combined.Freq);
            }

            return pq.Count > 0 ? pq.Dequeue() : null;
        }

        public static Dictionary<byte, string> GenerateHuffmanCodes(Node? root)
        {
            Dictionary<byte, string> codeDict = new();
            Traverse(root, "", codeDict);
            return codeDict;

            static void Traverse(Node? node, string code, Dictionary<byte, string> dict)
            {
                if (node == null) return;

                if (node.Byte.HasValue)
                {
                    dict[node.Byte.Value] = code;
                    return;
                }

                Traverse(node.Left, code + "0", dict);
                Traverse(node.Right, code + "1", dict);
            }
        }

        public static byte[] Compress(byte[] inputData, Dictionary<byte, string> codeDict, out int bitLength)
        {
            using var ms = new MemoryStream();
            int currentByte = 0;
            int bitCount = 0;
            bitLength = 0;

            foreach (var b in inputData)
            {
                string code = codeDict[b];
                foreach (char bitChar in code)
                {
                    currentByte <<= 1;
                    if (bitChar == '1')
                    {
                        currentByte |= 1;
                    }
                    bitCount++;
                    bitLength++;

                    if (bitCount == 8)
                    {
                        ms.WriteByte((byte)currentByte);
                        currentByte = 0;
                        bitCount = 0;
                    }
                }
            }

            // Siden vi ikke kan garantere at bit-sekvensen produsert av Huffman-kodene "går opp" perfekt i n antall byte, håndterer vi eventuelle bits som er til overs her
            if (bitCount > 0)
            {
                currentByte <<= 8 - bitCount;
                ms.WriteByte((byte)currentByte);
            }

            return ms.ToArray();
        }

        public static byte[] Decode(byte[] compressedBits, int bitLength, Node? huffmanTree)
        {
            if (huffmanTree == null || bitLength == 0) return Array.Empty<byte>();

            using var ms = new MemoryStream();
            var currentNode = huffmanTree;
            int processedBits = 0;

            foreach (byte b in compressedBits)
            {
                for (int i = 7; i >= 0; i--)
                {
                    if (processedBits >= bitLength) break;

                    int bit = (b >> i) & 1;
                    currentNode = (bit == 0) ? currentNode?.Left : currentNode?.Right;

                    if (currentNode != null && currentNode.Byte.HasValue)
                    {
                        ms.WriteByte(currentNode.Byte.Value);
                        currentNode = huffmanTree;
                    }

                    processedBits++;
                }
            }

            return ms.ToArray();
        }

        // Tar seg av hele prosessen fra input til output
        public static (byte[] encoded, int bitLength, Node? tree) Encode(byte[] data)
        {
            // 1. Bygger frekvens-map
            Dictionary<byte, int> freqDict = [];
            foreach (byte b in data)
            {
                freqDict[b] = freqDict.GetValueOrDefault(b, 0) + 1;
            }

            // 2. Bygger Huffman-treet
            Node? huffmanTree = BuildHuffmanTree(freqDict);

            // 3. Bygger Huffman-koder til bytes
            Dictionary<byte, string> huffmanCodes = GenerateHuffmanCodes(huffmanTree);

            // 4. Encode data into packed byte array
            byte[] encodedBytes = Compress(data, huffmanCodes, out int bitLength);

            return (encodedBytes, bitLength, huffmanTree);
        }
    }
}