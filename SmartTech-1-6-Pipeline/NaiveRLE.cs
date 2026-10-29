using System.IO;

namespace ST_1_6_Pipeline
{
    /*
    Naiv RLE-implementasjon som jobber fra og til byte[].
    */
    internal static class NaiveRLE
    {
        const int BYTE_SIZE = 256; // brukes for å begrense størrelsen på hver enkelt count slik at den får plass i en byte

        public static byte[] Encode(byte[] fileBytes)
        {
            if (fileBytes == null || fileBytes.Length == 0)
            {
                return [];
            }

            using MemoryStream ms = new();
            int i = 0;

            while (i < fileBytes.Length)
            {
                int runLength = 1;

                while (i + runLength < fileBytes.Length && fileBytes[i + runLength] == fileBytes[i] && runLength < BYTE_SIZE)
                {
                    runLength++;
                }

                ms.WriteByte((byte)(runLength-1));
                ms.WriteByte(fileBytes[i]);
                i += runLength;
            
            }

            return ms.ToArray();
        }

        public static byte[] Decode(byte[] encodedBytes)
        {
            if (encodedBytes == null || encodedBytes.Length == 0)
            {
                return [];
            }

            using MemoryStream ms = new();
            int i = 0;

            while (i < encodedBytes.Length)
            {
                int count = encodedBytes[i++] + 1;
                for (int c = 0; c < count; c++)
                {
                    ms.WriteByte(encodedBytes[i]);
                }
                i++;
            }

            return ms.ToArray();
        }
    }
}