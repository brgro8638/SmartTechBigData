using System.IO;

namespace ST_1_6_Pipeline
{
    internal class Huffman
    {
        public class Node(byte? b, int freq)
        {
            public byte? Byte { get; set; } = b;
            public int Freq { get; set; } = freq;
            public Node? Left { get; set; } = null;
            public Node? Right { get; set; } = null;
        }

        public class Result(byte[] e, int b, Node? t = null)
        {
            public byte[] encoded = e;
            public int bitLength = b;
            public Node? tree = t;
        }

        private static Dictionary<byte, int> BuildFrequencyTable(byte[] data)
        {
            Dictionary<byte, int> frequencyTable = new();
            foreach (byte b in data)
            {
                frequencyTable[b] = frequencyTable.GetValueOrDefault(b, 0) + 1;
            }
            return frequencyTable;
        }

        public static Node? BuildHuffmanTree(byte[] data)
        {
            Dictionary<byte, int> frequencyTable = BuildFrequencyTable(data);
            return BuildHuffmanTree(frequencyTable);
        }

        public static Node? BuildHuffmanTree(Dictionary<byte, int> frequencyTable)
        {
            PriorityQueue<Node, int> pq = new();

            foreach (KeyValuePair<byte, int> kvp in frequencyTable)
            {
                pq.Enqueue(new Node(kvp.Key, kvp.Value), kvp.Value);
            }

            while (pq.Count > 1)
            {
                Node left = pq.Dequeue();
                Node right = pq.Dequeue();

                Node combined = new(null, left.Freq + right.Freq)
                {
                    Left = left,
                    Right = right
                };

                pq.Enqueue(combined, combined.Freq);
            }

            return pq.Count > 0 ? pq.Dequeue() : null;
        }

        public static Dictionary<byte, string> BuildHuffmanCodebook(Node? root)
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
            using MemoryStream ms = new();
            int currentByte = 0;
            int bitCount = 0;
            bitLength = 0;

            foreach (byte b in inputData)
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

        public static byte[] Decompress(byte[] compressedBits, int bitLength, Node? huffmanTree)
        {
            if (huffmanTree == null || bitLength == 0) 
            {
                return [];
            }

            using MemoryStream ms = new();
            Node? currentNode = huffmanTree;
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
        public static Result Encode(byte[] data)
        {
            // 1. Bygger frekvens-map
            Dictionary<byte, int> frequencyTable = BuildFrequencyTable(data);

            // 2. Bygger Huffman-treet
            Node? tree = BuildHuffmanTree(frequencyTable);

            // 3. Bygger Huffman-koder
            Dictionary<byte, string> codeBook = BuildHuffmanCodebook(tree);

            // 4. Bygger byte array basert på sekvensen av Huffman-koder
            byte[] encodedBytes = Compress(data, codeBook, out int bitLength);

            return new Result(encodedBytes, bitLength, tree);
        }

        // Fullfører prosessen når treet og kodene allerede er konstruert
        public static Result Encode(byte[] data, Node? tree, Dictionary<byte, string> codeBook)
        {
            byte[] encodedBytes = Compress(data, codeBook, out int bitLength);

            return new Result(encodedBytes, bitLength);
        }
    }
}