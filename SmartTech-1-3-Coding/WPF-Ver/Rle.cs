using System.IO;

namespace ST_1_3_Coding
{
    public static class RLE
    {
        public static byte[] Encode(byte[] input)
        {
            if (input == null || input.Length == 0)
                return [];

            using MemoryStream ms = new();
            int i = 0;

            while (i < input.Length)
            {
                // identifiserer et "run" bestående av repeterte bytes (bryr oss bare ved runlength 3+, pga header)
                int runLength = 1;
                while (i + runLength < input.Length && input[i + runLength] == input[i] && runLength < 128)
                {
                    runLength++;
                }

                if (runLength >= 3)
                {
                    // negativ header "flagger" et run
                    sbyte header = (sbyte)-(runLength - 1);
                    ms.WriteByte((byte)header);
                    ms.WriteByte(input[i]);
                    i += runLength;
                }
                else
                {
                    // fant vi ikke repeterte bytes prøver vi å skrive ikke-repeterte bytes direkte til output (i bolker)
                    int literalStart = i;
                    int literalLength = 0;

                    while (i < input.Length && literalLength < 128) // maks bolk-lengde er 128 fordi vi må reservere negative halvdel av sbyte til run-flagging
                    {
                        // stopper dersom vi møter slutten av array/bolk eller starten på et nytt "run"
                        if (i + 2 < input.Length &&
                            input[i] == input[i + 1] &&
                            input[i] == input[i + 2])
                        {
                            break;
                        }

                        literalLength++;
                        i++;
                    }

                    // positiv header "flagger" en seksjon uten repetisjon
                    sbyte header = (sbyte)(literalLength - 1);
                    ms.WriteByte((byte)header);
                    ms.Write(input, literalStart, literalLength);
                }
            }

            return ms.ToArray();
        }

        public static byte[] Decode(byte[] compressed)
        {
            if (compressed == null || compressed.Length == 0)
                return [];

            using var ms = new MemoryStream();
            int i = 0;

            while (i < compressed.Length)
            {
                sbyte header = (sbyte)compressed[i++];

                if (header >= 0) // positiv header betyr ikke-repeterte bytes
                {
                    // vi skriver gitt antall bytes direkte til output
                    int count = header + 1;
                    ms.Write(compressed, i, count);
                    i += count;
                }
                else if (header != -128) // negativ header betyr repeterte bytes
                {
                    // vi skriver gitt antall kopier av samme byte til output
                    int count = 1 - header;
                    byte val = compressed[i++];
                    for (int c = 0; c < count; c++)
                    {
                        ms.WriteByte(val);
                    }
                }
            }

            return ms.ToArray();
        }
    }
}