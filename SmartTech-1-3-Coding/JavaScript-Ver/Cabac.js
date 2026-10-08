class Cabac {
    // 32-bit arithmetic precision bounds (using BigInt to prevent 32-bit signed bitwise shifts)
    static TOP_VALUE = 0xFFFFFFFFn;
    static FIRST_QTR = 0x40000000n;
    static HALF      = 0x80000000n;
    static THIRD_QTR = 0xC0000000n;

    /**
     * Encodes a Uint8Array using Context-Adaptive Binary Arithmetic Coding.
     * @param {Uint8Array} inputData
     * @returns {Uint8Array}
     */
    static encode(inputData) {
        if (!inputData || inputData.length === 0) return new Uint8Array(0);

        const bitWriter = new CabacBitWriter();
        
        // Header: Write original data length (32-bit little-endian integer)
        bitWriter.writeInt32(inputData.length);

        const contexts = this._initContexts();

        let low = 0n;
        let high = this.TOP_VALUE;
        let bitsToFollow = 0;

        for (let i = 0; i < inputData.length; i++) {
            const symbol = inputData[i];

            // Binarize byte into 8 bits (MSB first)
            for (let bitIdx = 7; bitIdx >= 0; bitIdx--) {
                const bit = (symbol >> bitIdx) & 1;
                const ctx = contexts[7 - bitIdx]; // Context indexed by bit position

                // 1. Calculate interval range and split based on context probability
                const range = high - low + 1n;
                const mid = low + (range * BigInt(ctx.count0) / BigInt(ctx.total)) - 1n;

                if (bit === 0) {
                    high = mid;
                } else {
                    low = mid + 1n;
                }

                // 2. Update context model (Adaptation)
                ctx.update(bit);

                // 3. Renormalization
                while (true) {
                    if (high < this.HALF) {
                        bitWriter.writeBit(0);
                        for (; bitsToFollow > 0; bitsToFollow--) bitWriter.writeBit(1);
                    } else if (low >= this.HALF) {
                        bitWriter.writeBit(1);
                        for (; bitsToFollow > 0; bitsToFollow--) bitWriter.writeBit(0);
                        low -= this.HALF;
                        high -= this.HALF;
                    } else if (low >= this.FIRST_QTR && high < this.THIRD_QTR) {
                        bitsToFollow++;
                        low -= this.FIRST_QTR;
                        high -= this.FIRST_QTR;
                    } else {
                        break;
                    }

                    low = (low << 1n) & 0xFFFFFFFFn;
                    high = ((high << 1n) | 1n) & 0xFFFFFFFFn;
                }
            }
        }

        // Flush remaining arithmetic engine bits
        bitsToFollow++;
        if (low < this.FIRST_QTR) {
            bitWriter.writeBit(0);
            while (bitsToFollow > 0) { bitWriter.writeBit(1); bitsToFollow--; }
        } else {
            bitWriter.writeBit(1);
            while (bitsToFollow > 0) { bitWriter.writeBit(0); bitsToFollow--; }
        }

        bitWriter.flush();
        return bitWriter.toUint8Array();
    }

    /**
     * Decodes CABAC-compressed bytes back into original uncompressed Uint8Array.
     * @param {Uint8Array} compressedBytes
     * @returns {Uint8Array}
     */
    static decode(compressedBytes) {
        if (!compressedBytes || compressedBytes.length === 0) return new Uint8Array(0);

        const bitReader = new CabacBitReader(compressedBytes);

        const originalLength = bitReader.readInt32();
        const contexts = this._initContexts();

        // Prefill 32-bit value register
        let value = 0n;
        for (let i = 0; i < 32; i++) {
            value = (value << 1n) | BigInt(bitReader.readBit());
        }

        let low = 0n;
        let high = this.TOP_VALUE;
        const outputData = new Uint8Array(originalLength);

        for (let i = 0; i < originalLength; i++) {
            let decodedByte = 0;

            for (let bitIdx = 7; bitIdx >= 0; bitIdx--) {
                const contextModel = contexts[7 - bitIdx];

                const range = high - low + 1n;
                const mid = low + (range * BigInt(contextModel.count0) / BigInt(contextModel.total)) - 1n;

                let bit;
                if (value <= mid) {
                    bit = 0;
                    high = mid;
                } else {
                    bit = 1;
                    low = mid + 1n;
                }

                decodedByte |= (bit << bitIdx);

                // Update context model (Adaptation)
                contextModel.update(bit);

                // Renormalization
                while (true) {
                    if (high < this.HALF) {
                        // No adjust
                    } else if (low >= this.HALF) {
                        value -= this.HALF;
                        low -= this.HALF;
                        high -= this.HALF;
                    } else if (low >= this.FIRST_QTR && high < this.THIRD_QTR) {
                        value -= this.FIRST_QTR;
                        low -= this.FIRST_QTR;
                        high -= this.FIRST_QTR;
                    } else {
                        break;
                    }

                    low = (low << 1n) & 0xFFFFFFFFn;
                    high = ((high << 1n) | 1n) & 0xFFFFFFFFn;
                    value = ((value << 1n) | BigInt(bitReader.readBit())) & 0xFFFFFFFFn;
                }
            }

            outputData[i] = decodedByte;
        }

        return outputData;
    }

    static _initContexts() {
        const contexts = new Array(8);
        for (let i = 0; i < 8; i++) contexts[i] = new ContextModel();
        return contexts;
    }
}

class ContextModel {
    constructor() {
        this.count0 = 1;
        this.count1 = 1;
    }

    get total() {
        return this.count0 + this.count1;
    }

    update(bit) {
        if (bit === 0) this.count0++;
        else this.count1++;

        if (this.total >= 2000) {
            this.count0 = (this.count0 >> 1) | 1;
            this.count1 = (this.count1 >> 1) | 1;
        }
    }
}

/*
class BitWriter {
    constructor() {
        this.bytes = [];
        this.currentByte = 0;
        this.bitCount = 0;
    }

    writeInt32(value) {
        // Writes 32-bit integer in Little-Endian format to match C# BinaryWriter
        this.bytes.push(
            value & 0xFF,
            (value >> 8) & 0xFF,
            (value >> 16) & 0xFF,
            (value >> 24) & 0xFF
        );
    }

    writeBit(bit) {
        this.currentByte = (this.currentByte << 1) | (bit & 1);
        this.bitCount++;
        if (this.bitCount === 8) {
            this.bytes.push(this.currentByte);
            this.currentByte = 0;
            this.bitCount = 0;
        }
    }

    flush() {
        if (this.bitCount > 0) {
            this.currentByte <<= (8 - this.bitCount);
            this.bytes.push(this.currentByte);
            this.bitCount = 0;
            this.currentByte = 0;
        }
    }

    toUint8Array() {
        return new Uint8Array(this.bytes);
    }
}

class BitReader {
    constructor(uint8Array) {
        this.data = uint8Array;
        this.offset = 0;
        this.currentByte = 0;
        this.bitCount = 0;
    }

    readInt32() {
        // Reads 32-bit integer in Little-Endian format to match C# BinaryReader
        const b0 = this.data[this.offset++];
        const b1 = this.data[this.offset++];
        const b2 = this.data[this.offset++];
        const b3 = this.data[this.offset++];
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }

    readBit() {
        if (this.bitCount === 0) {
            this.currentByte = this.offset < this.data.length ? this.data[this.offset++] : 0;
            this.bitCount = 8;
        }
        this.bitCount--;
        return (this.currentByte >> this.bitCount) & 1;
    }
}
*/