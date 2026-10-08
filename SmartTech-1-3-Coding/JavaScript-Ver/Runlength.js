class Rle {
    /**
     * koder byte array med RLE optimisert for å forminske ekspansjon
     * @param {Uint8Array} input 
     * @returns {Uint8Array}
     */
    static encode(input) {
        if (!input || input.length === 0) 
        {
            return new Uint8Array(0);
        }

        const output = [];
        let i = 0;

        while (i < input.length) 
        {
            // Identifiserer et "run" bestående av repeterte bytes (bryr oss bare ved runlength 3+, pga header)
            let runLength = 1;
            while (i + runLength < input.length && input[i + runLength] === input[i] &&runLength < 128) 
            {
                runLength++;
            }

            if (runLength >= 3) 
            {
                // Negativ header "flagger" et run (bruker bitwise & 0xFF for å konvertere sbyte til uint8)
                const header = (-(runLength - 1)) & 0xFF;
                output.push(header);
                output.push(input[i]);
                i += runLength;
            } 
            else 
            {
                // Fant vi ikke repeterte bytes prøver vi å skrive ikke-repeterte bytes direkte til output (i bolker)
                const literalStart = i;
                let literalLength = 0;

                while (i < input.length && literalLength < 128) 
                {
                    // Stopper dersom vi møter slutten av array/bolk eller starten på et nytt "run"
                    if (i + 2 < input.length && input[i] === input[i + 1] && input[i] === input[i + 2]) 
                    {
                        break;
                    }

                    literalLength++;
                    i++;
                }

                // Positiv header "flagger" en seksjon uten repetisjon
                const header = literalLength - 1;
                output.push(header);
                
                for (let j = 0; j < literalLength; j++) 
                {
                    output.push(input[literalStart + j]);
                }
            }
        }

        return new Uint8Array(output);
    }

    /**
     * dekoder komprimert byte array
     * @param {Uint8Array} compressed 
     * @returns {Uint8Array}
     */
    static decode(compressed) {
        if (!compressed || compressed.length === 0) 
        {
            return new Uint8Array(0);
        }

        const output = [];
        let i = 0;

        while (i < compressed.length) 
        {
            // Konverterer ubehandlet uint8 byte til signert sbyte (-128 til 127)
            let header = compressed[i++];
            if (header > 127) 
            {
                header -= 256;
            }

            if (header >= 0) {
                // Positiv header betyr ikke-repeterte bytes
                const count = header + 1;
                for (let j = 0; j < count; j++) 
                {
                    output.push(compressed[i + j]);
                }
                i += count;
            } 
            else if (header !== -128) 
            {
                // Negativ header betyr repeterte bytes
                const count = 1 - header;
                const val = compressed[i++];
                for (let c = 0; c < count; c++) 
                {
                    output.push(val);
                }
            }
        }

        return new Uint8Array(output);
    }
}