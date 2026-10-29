using System.IO;

namespace ST_1_6_Pipeline
{
    internal class Cabac
    {
        // oppdeling av fullstendig range innen 32-bit integer
        private const uint TopValue = 0xFFFFFFFFu;
        private const uint FirstQtr = 0x40000000u;
        private const uint Half     = 0x80000000u;
        private const uint ThirdQtr = 0xC0000000u;

        public static byte[] Encode(byte[] inputData)
        {
            if (inputData == null || inputData.Length == 0) return [];

            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);

            // original lengde skrives til header (nødvendig ved decoding)
            writer.Write(inputData.Length);

            BitWriter bitWriter = new(writer);
            ContextModel[] contexts = InitContexts(); 

            uint low = 0;
            uint high = TopValue;
            int bitsToFollow = 0;

            foreach (byte symbol in inputData)
            {
                // binarisering
                for (int bitIdx = 7; bitIdx >= 0; bitIdx--)
                {
                    int bit = (symbol >> bitIdx) & 1;
                    ContextModel contextModel = contexts[7 - bitIdx];

                    // setter intervall vha context
                    ulong range = (ulong)high - low + 1;
                    uint mid = low + (uint)(range * contextModel.Count0 / contextModel.Total) - 1;

                    if (bit == 0)
                    {
                        high = mid;
                    }
                    else
                    {
                        low = mid + 1;
                    }

                    // oppdaterer context
                    contextModel.Update(bit);

                    // renormalisering
                    while (true)
                    {
                        if (high < Half)
                        {
                            bitWriter.WriteBit(0);
                            for (; bitsToFollow > 0; bitsToFollow--) bitWriter.WriteBit(1);
                        }
                        else if (low >= Half)
                        {
                            bitWriter.WriteBit(1);
                            for (; bitsToFollow > 0; bitsToFollow--) bitWriter.WriteBit(0);
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
            }

            // "flusher" bits som er til overs til full byte
            bitsToFollow++;
            if (low < FirstQtr)
            {
                bitWriter.WriteBit(0);
                while (bitsToFollow > 0) { bitWriter.WriteBit(1); bitsToFollow--; }
            }
            else
            {
                bitWriter.WriteBit(1);
                while (bitsToFollow > 0) { bitWriter.WriteBit(0); bitsToFollow--; }
            }

            bitWriter.Flush();
            return ms.ToArray();
        }

        public static byte[] Decode(byte[] compressedBytes)
        {
            if (compressedBytes == null || compressedBytes.Length == 0)
            {
                return [];
            }

            using MemoryStream ms = new(compressedBytes);
            using BinaryReader reader = new(ms);

            int originalLength = reader.ReadInt32();
            BitReader bitReader = new(reader);
            ContextModel[] contexts = InitContexts();

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
                byte decodedByte = 0;

                for (int bitIdx = 7; bitIdx >= 0; bitIdx--)
                {
                    ContextModel contextModel = contexts[7 - bitIdx];

                    ulong range = (ulong)high - low + 1;
                    uint mid = low + (uint)(range * contextModel.Count0 / contextModel.Total) - 1;

                    int bit;
                    if (value <= mid)
                    {
                        bit = 0;
                        high = mid;
                    }
                    else
                    {
                        bit = 1;
                        low = mid + 1;
                    }

                    decodedByte |= (byte)(bit << bitIdx);
                    contextModel.Update(bit);

                    // renormalisering
                    while (true)
                    {
                        if (high < Half)
                        {
                            // pass
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

                outputData[i] = decodedByte;
            }

            return outputData;
        }

        #region Context Model & Helper Classes

        private class ContextModel
        {
            public uint Count0 { get; private set; } = 1;
            public uint Count1 { get; private set; } = 1;
            public uint Total => Count0 + Count1;

            public void Update(int bit)
            {
                if (bit == 0) Count0++;
                else Count1++;
        
                // skalerer ned counts når totalen blir høy, slik at nyere kontekst veier tyngre (adaptiv)
                if (Total >= 2000)
                {
                    Count0 = (Count0 >> 1) | 1;
                    Count1 = (Count1 >> 1) | 1;
                }
            }
        }

        private static ContextModel[] InitContexts()
        {
            var contexts = new ContextModel[8];
            for (int i = 0; i < 8; i++) contexts[i] = new ContextModel();
            return contexts;
        }

        private class BitWriter
        {
            private readonly BinaryWriter _writer;
            private byte _currentByte;
            private int _bitCount;

            public BitWriter(BinaryWriter writer) => _writer = writer;

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
                }
            }
        }

        private class BitReader
        {
            private readonly BinaryReader _reader;
            private byte _currentByte;
            private int _bitCount;

            public BitReader(BinaryReader reader) => _reader = reader;

            public uint ReadBit()
            {
                if (_bitCount == 0)
                {
                    _currentByte = _reader.BaseStream.Position < _reader.BaseStream.Length ? _reader.ReadByte() : (byte)0;
                    _bitCount = 8;
                }
                _bitCount--;
                return (uint)((_currentByte >> _bitCount) & 1);
            }
        }

        #endregion
    }
}