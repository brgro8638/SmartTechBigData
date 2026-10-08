// Arithmetic

class BitWriter {
    constructor(outputBytes) {
        this.outputBytes = outputBytes;
        this.currentByte = 0;
        this.bitCount = 0;
    }

    writeBit(bit) {
        this.currentByte = ((this.currentByte << 1) | (bit & 1)) & 0xFF;
        this.bitCount++;

        if (this.bitCount === 8) {
            this.outputBytes.push(this.currentByte);
            this.currentByte = 0;
            this.bitCount = 0;
        }
    }

    flush() {
        if (this.bitCount > 0) {
            this.currentByte <<= (8 - this.bitCount);
            this.outputBytes.push(this.currentByte & 0xFF);
            this.currentByte = 0;
            this.bitCount = 0;
        }
    }
}

class BitReader {
    constructor(buffer, offset = 0) {
        this.buffer = buffer;
        this.offset = offset;
        this.currentByte = 0;
        this.bitCount = 0;
    }

    readBit() {
        if (this.bitCount === 0) {
            if (this.offset < this.buffer.length) {
                this.currentByte = this.buffer[this.offset++];
            } else {
                this.currentByte = 0;
            }
            this.bitCount = 8;
        }

        this.bitCount--;
        return (this.currentByte >> this.bitCount) & 1;
    }
}

// CABAC

class CabacBitWriter {
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

class CabacBitReader {
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

// Arithmetic Preview

/**
 * Converts a compressed byte array into its exact floating-point decimal string representation [0, 1).
 * @param {Uint8Array} compressedBytes - Output from CABAC/arithmetic encoder
 * @param {number} precisionDigits - Number of decimal places to calculate (default 20)
 * @returns {string} - String formatted as "0.xxxxxxxx..."
 */
function byteStreamToFloatString(compressedBytes, precisionDigits = 32) {
    if (!compressedBytes || compressedBytes.length === 0) return "0.0";

    // Calculate how many bytes we need: ~3.32 bits per decimal digit + safety margin
    const neededBits = Math.ceil(precisionDigits * 3.3219) + 16;
    const neededBytes = Math.min(compressedBytes.length, Math.ceil(neededBits / 8));

    let num = 0n;
    for (let i = 0; i < neededBytes; i++) {
        num = (num << 8n) | BigInt(compressedBytes[i]);
    }

    const totalBits = BigInt(neededBytes * 8);
    const base10Multiplier = 10n ** BigInt(precisionDigits);
    
    let decimalPart = ((num * base10Multiplier) >> totalBits).toString();
    decimalPart = decimalPart.padStart(precisionDigits, '0');

    return `0.${decimalPart}`;
}