using System.IO;

namespace ST_1_3_Coding
{
    public class Arithmetic
    {
        // 32-bit unsigned integer arithmetic precision bounds
        private const uint TopValue = 0xFFFFFFFFu;
        private const uint FirstQtr = 0x40000000u;
        private const uint Half     = 0x80000000u;
        private const uint ThirdQtr = 0xC0000000u;

        /// <summary>
        /// Encodes a byte array using 256-symbol static frequency Arithmetic Coding.
        /// </summary>
        public static byte[] Encode(byte[] inputData)
        {
            if (inputData == null || inputData.Length == 0)
                return [];

            // 1. Calculate frequency table (256 symbols)
            uint[] frequencies = new uint[256];
            foreach (byte b in inputData)
            {
                frequencies[b]++;
            }

            // Ensure every symbol has a non-zero count to avoid 0-width intervals
            for (int i = 0; i < 256; i++)
            {
                if (frequencies[i] == 0) frequencies[i] = 1;
            }

            // 2. Build Cumulative Frequency Distribution
            uint[] cumFreq = new uint[257];
            cumFreq[0] = 0;
            for (int i = 0; i < 256; i++)
            {
                cumFreq[i + 1] = cumFreq[i] + frequencies[i];
            }
            uint totalFreq = cumFreq[256];

            // 3. Encode data
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);

            // Write metadata header (Original length + Frequency table)
            writer.Write(inputData.Length);
            for (int i = 0; i < 256; i++)
            {
                writer.Write(frequencies[i]);
            }

            uint low = 0;
            uint high = TopValue;
            int bitsToFollow = 0;
            var bitBuffer = new BitWriter(writer);

            foreach (byte symbol in inputData)
            {
                ulong range = (ulong)high - low + 1;
                high = low + (uint)(range * cumFreq[symbol + 1] / totalFreq) - 1;
                low += (uint)(range * cumFreq[symbol] / totalFreq);

                // Renormalization
                while (true)
                {
                    if (high < Half)
                    {
                        bitBuffer.WriteBit(0);
                        for (; bitsToFollow > 0; bitsToFollow--) bitBuffer.WriteBit(1);
                    }
                    else if (low >= Half)
                    {
                        bitBuffer.WriteBit(1);
                        for (; bitsToFollow > 0; bitsToFollow--) bitBuffer.WriteBit(0);
                        low -= Half;
                        high -= Half;
                    }
                    else if (low >= FirstQtr && high < ThirdQtr)
                    {
                        bitsToFollow++;
                        low -= FirstQtr;
                        high -= FirstQtr;
                    }
                    else
                    {
                        break;
                    }

                    low <<= 1;
                    high = (high << 1) | 1;
                }
            }

            // Flush final state bits
            bitsToFollow++;
            if (low < FirstQtr)
            {
                bitBuffer.WriteBit(0);
                while (bitsToFollow > 0)
                {
                    bitBuffer.WriteBit(1);
                    bitsToFollow--;
                }
            }
            else
            {
                bitBuffer.WriteBit(1);
                while (bitsToFollow > 0)
                {
                    bitBuffer.WriteBit(0);
                    bitsToFollow--;
                }
            }

            bitBuffer.Flush();
            return ms.ToArray();
        }

        /// <summary>
        /// Decodes an arithmetic-compressed byte array back to its original state.
        /// </summary>
        public static byte[] Decode(byte[] compressedBytes)
        {
            if (compressedBytes == null || compressedBytes.Length == 0)
                return [];

            using var ms = new MemoryStream(compressedBytes);
            using var reader = new BinaryReader(ms);

            // 1. Read metadata header
            int originalLength = reader.ReadInt32();
            uint[] frequencies = new uint[256];
            for (int i = 0; i < 256; i++)
            {
                frequencies[i] = reader.ReadUInt32();
            }

            // 2. Rebuild Cumulative Frequency Distribution
            uint[] cumFreq = new uint[257]; // 256 + <EOS>
            cumFreq[0] = 0;
            for (int i = 0; i < 256; i++)
            {
                cumFreq[i + 1] = cumFreq[i] + frequencies[i];
            }
            uint totalFreq = cumFreq[256];

            var bitReader = new BitReader(reader);

            // Fill initial 32-bit register value
            uint value = 0;
            for (int i = 0; i < 32; i++)
            {
                value = (value << 1) | bitReader.ReadBit();
            }

            uint low = 0;
            uint high = TopValue;
            byte[] outputData = new byte[originalLength];

            for (int i = 0; i < originalLength; i++)
            {
                ulong range = (ulong)high - low + 1;
                uint cum = (uint)((((ulong)value - low + 1) * totalFreq - 1) / range);

                // Binary search or linear scan for matching symbol in cumFreq table
                byte symbol = 0;
                while (cumFreq[symbol + 1] <= cum)
                {
                    symbol++;
                }

                outputData[i] = symbol;

                high = low + (uint)(range * cumFreq[symbol + 1] / totalFreq) - 1;
                low += (uint)(range * cumFreq[symbol] / totalFreq);

                // Renormalization
                while (true)
                {
                    if (high < Half)
                    {
                        // No adjustment
                    }
                    else if (low >= Half)
                    {
                        value -= Half;
                        low -= Half;
                        high -= Half;
                    }
                    else if (low >= FirstQtr && high < ThirdQtr)
                    {
                        value -= FirstQtr;
                        low -= FirstQtr;
                        high -= FirstQtr;
                    }
                    else
                    {
                        break;
                    }

                    low <<= 1;
                    high = (high << 1) | 1;
                    value = (value << 1) | bitReader.ReadBit();
                }
            }

            return outputData;
        }

        #region Binary Helpers

        private class BitWriter
        {
            private readonly BinaryWriter _writer;
            private byte _currentByte;
            private int _bitCount;

            public BitWriter(BinaryWriter writer)
            {
                _writer = writer;
            }

            public void WriteBit(uint bit)
            {
                _currentByte = (byte)(((uint)_currentByte << 1) | (bit & 1));
                _bitCount++;

                if (_bitCount == 8)
                {
                    _writer.Write(_currentByte);
                    _currentByte = 0;
                    _bitCount = 0;
                }
            }

            public void Flush()
            {
                if (_bitCount > 0)
                {
                    _currentByte <<= 8 - _bitCount;
                    _writer.Write(_currentByte);
                    _currentByte = 0;
                    _bitCount = 0;
                }
            }
        }

        private class BitReader
        {
            private readonly BinaryReader _reader;
            private byte _currentByte;
            private int _bitCount;

            public BitReader(BinaryReader reader)
            {
                _reader = reader;
            }

            public uint ReadBit()
            {
                if (_bitCount == 0)
                {
                    if (_reader.BaseStream.Position < _reader.BaseStream.Length)
                    {
                        _currentByte = _reader.ReadByte();
                    }
                    else
                    {
                        _currentByte = 0;
                    }
                    _bitCount = 8;
                }

                _bitCount--;
                return (uint)((_currentByte >> _bitCount) & 1);
            }
        }

        #endregion
    }
}