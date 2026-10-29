const TOP_VALUE = 0xFFFFFFFF;
const FIRST_QTR = 0x40000000;
const HALF      = 0x80000000;
const THIRD_QTR = 0xC0000000;

class Arithmetic {
    /**
     * Encodes a byte array (Uint8Array) using 256-symbol static frequency Arithmetic Coding.
     * @param {Uint8Array} inputData 
     * @returns {Uint8Array}
     */
    static encode(inputData) {
        if (!inputData || inputData.length === 0) {
            return new Uint8Array(0);
        }

        // 1. Calculate frequency table (256 symbols)
        const frequencies = new Uint32Array(256);
        for (const b of inputData) {
            frequencies[b]++;
        }

        // Ensure every symbol has a non-zero count to avoid 0-width intervals
        for (let i = 0; i < 256; i++) {
            if (frequencies[i] === 0) frequencies[i] = 1;
        }

        // 2. Build Cumulative Frequency Distribution
        const cumFreq = new Uint32Array(257);
        cumFreq[0] = 0;
        for (let i = 0; i < 256; i++) {
            cumFreq[i + 1] = cumFreq[i] + frequencies[i];
        }
        const totalFreq = cumFreq[256];

        // 3. Encode data
        const outputBytes = [];

        // Write metadata header (Original length + Frequency table) as little-endian 32-bit integers
        const headerBuffer = new ArrayBuffer(4 + 256 * 4);
        const headerView = new DataView(headerBuffer);
        
        headerView.setInt32(0, inputData.length, true);
        for (let i = 0; i < 256; i++) {
            headerView.setUint32(4 + i * 4, frequencies[i], true);
        }
        
        outputBytes.push(...new Uint8Array(headerBuffer));

        let low = 0;
        let high = TOP_VALUE;
        let bitsToFollow = 0;
        const bitBuffer = new BitWriter(outputBytes);

        for (const symbol of inputData) {
            const range = BigInt(high - low + 1);
            
            const highOffset = Number((range * BigInt(cumFreq[symbol + 1])) / BigInt(totalFreq)) - 1;
            const lowOffset = Number((range * BigInt(cumFreq[symbol])) / BigInt(totalFreq));

            high = (low + highOffset) >>> 0;
            low = (low + lowOffset) >>> 0;

            // Renormalization
            while (true) {
                if (high < HALF) {
                    bitBuffer.writeBit(0);
                    for (; bitsToFollow > 0; bitsToFollow--) bitBuffer.writeBit(1);
                } else if (low >= HALF) {
                    bitBuffer.writeBit(1);
                    for (; bitsToFollow > 0; bitsToFollow--) bitBuffer.writeBit(0);
                    low = (low - HALF) >>> 0;
                    high = (high - HALF) >>> 0;
                } else if (low >= FIRST_QTR && high < THIRD_QTR) {
                    bitsToFollow++;
                    low = (low - FIRST_QTR) >>> 0;
                    high = (high - FIRST_QTR) >>> 0;
                } else {
                    break;
                }

                low = (low << 1) >>> 0;
                high = ((high << 1) | 1) >>> 0;
            }
        }

        // Flush final state bits
        bitsToFollow++;
        if (low < FIRST_QTR) {
            bitBuffer.writeBit(0);
            while (bitsToFollow > 0) {
                bitBuffer.writeBit(1);
                bitsToFollow--;
            }
        } else {
            bitBuffer.writeBit(1);
            while (bitsToFollow > 0) {
                bitBuffer.writeBit(0);
                bitsToFollow--;
            }
        }

        bitBuffer.flush();
        return Uint8Array.from(outputBytes);
    }

    /**
     * Decodes an arithmetic-compressed byte array back to its original state.
     * @param {Uint8Array} compressedBytes 
     * @returns {Uint8Array}
     */
    static decode(compressedBytes) {
        if (!compressedBytes || compressedBytes.length === 0) {
            return new Uint8Array(0);
        }

        const dataView = new DataView(compressedBytes.buffer, compressedBytes.byteOffset, compressedBytes.byteLength);

        // 1. Read metadata header
        const originalLength = dataView.getInt32(0, true);
        const frequencies = new Uint32Array(256);
        for (let i = 0; i < 256; i++) {
            frequencies[i] = dataView.getUint32(4 + i * 4, true);
        }

        // 2. Rebuild Cumulative Frequency Distribution
        const cumFreq = new Uint32Array(257);
        cumFreq[0] = 0;
        for (let i = 0; i < 256; i++) {
            cumFreq[i + 1] = cumFreq[i] + frequencies[i];
        }
        const totalFreq = cumFreq[256];

        const bitReader = new BitReader(compressedBytes, 4 + 256 * 4);

        // Fill initial 32-bit register value
        let value = 0;
        for (let i = 0; i < 32; i++) {
            value = ((value << 1) | bitReader.readBit()) >>> 0;
        }

        let low = 0;
        let high = TOP_VALUE;
        const outputData = new Uint8Array(originalLength);

        for (let i = 0; i < originalLength; i++) {
            const range = BigInt(high - low + 1);
            const num = (BigInt(value - low + 1) * BigInt(totalFreq) - 1n) / range;
            const cum = Number(num) >>> 0;

            // Search for matching symbol in cumFreq table
            let symbol = 0;
            while (cumFreq[symbol + 1] <= cum) {
                symbol++;
            }

            outputData[i] = symbol;

            const highOffset = Number((range * BigInt(cumFreq[symbol + 1])) / BigInt(totalFreq)) - 1;
            const lowOffset = Number((range * BigInt(cumFreq[symbol])) / BigInt(totalFreq));

            high = (low + highOffset) >>> 0;
            low = (low + lowOffset) >>> 0;

            // Renormalization
            while (true) {
                if (high < HALF) {
                    // pass
                } else if (low >= HALF) {
                    value = (value - HALF) >>> 0;
                    low = (low - HALF) >>> 0;
                    high = (high - HALF) >>> 0;
                } else if (low >= FIRST_QTR && high < THIRD_QTR) {
                    value = (value - FIRST_QTR) >>> 0;
                    low = (low - FIRST_QTR) >>> 0;
                    high = (high - FIRST_QTR) >>> 0;
                } else {
                    break;
                }

                low = (low << 1) >>> 0;
                high = ((high << 1) | 1) >>> 0;
                value = ((value << 1) | bitReader.readBit()) >>> 0;
            }
        }

        return outputData;
    }
}