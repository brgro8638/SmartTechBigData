# Video script: Data Coder (about 7 minutes)

Before recording: start the app with `streamlit run app.py`, run the benchmark tab once
so the results are ready, and open the code in an editor. Do not commit the video.

The CT slices are patient data. The app shows only pixels, but check that no patient
information is visible in the frame, and check the course rules on showing these images.

Timings on screen will differ a little from the numbers below, because they depend on the
machine. Sizes will match exactly at the same settings.

## 1. The task (0:00 to 0:30)

**Show:** the app, Input tab.

> This is my Data Coder. It reads image data in different ways and compares three entropy
> coders: Huffman, arithmetic coding and CABAC. It shows both how each one encodes the data
> and how well it performs. I wrote all three coders from scratch. Every run is decoded again
> from the bytes alone and compared with the original, so all the results are verified lossless.

## 2. Reading the data (0:30 to 1:30)

**Show:** click through the four source types in the sidebar. Then `sources.py`.

> There are four kinds of input. Generated images, where I control the statistics. Medical
> data: the Harnverhalt CT, read from DICOM. Real-life photos. And raw files, where any file
> can be read as 8- or 16-bit samples.
>
> Whatever the source, the loader returns a 2D integer array and its nominal bit depth.
> The CT can be read two ways: the stored 12-bit values, which are the full data, or an 8-bit
> soft-tissue window, which is what we look at on screen but clips everything outside the window.

**Show:** switch prediction between None and MED, point at the residual image and the right-hand histogram.

> Before coding I can apply prediction. MED is the predictor from JPEG-LS. The residual image
> is mostly gray, meaning zero, and the histogram gets narrow. That lowers the entropy, here
> from 6.24 to 4.49 bits per symbol. All coders get the same symbols, so any difference
> comes from the coders themselves.

## 3. Huffman (1:30 to 2:30)

**Show:** Huffman tab, code table. Then `code_lengths` and `canonical_codes` in `coders/huffman.py`.

> Huffman repeatedly merges the two rarest groups, and every merge adds one bit to all their
> symbols. Frequent symbols get short codes. I use canonical codes, so the header only needs
> the code lengths.
>
> Look at the "waste" column. A symbol with an ideal cost of 4.7 bits has to use 4 or 5. Huffman
> can only spend whole bits, and it can never use less than one bit per symbol.

## 4. Arithmetic coding (2:30 to 3:30)

**Show:** Arithmetic tab, the interval table and the chart. Then the encode loop in `coders/arithmetic.py`.

> Arithmetic coding gives the whole message one number. Each symbol narrows an interval in
> proportion to its probability. Here you can see low and high narrowing step by step. When
> their leading bits agree, those bits can never change, so they are sent and the interval
> is doubled. Pending bits handle the case where the interval straddles the middle.
>
> The chart shows why this matters: the output follows the information, regardless of symbol
> boundaries. So a symbol can cost a fraction of a bit. The price is that the model has to be
> sent in the header, and it does more work per symbol.

## 5. CABAC (3:30 to 5:00)

**Show:** CABAC tab, the bin table. Then `coders/cabac.py`: `encode_bin`, then `encode_residual`, then `_magnitude_class`.

> CABAC stands for Context-based Adaptive Binary Arithmetic Coding, and the name lists the
> parts. First binarization: each symbol becomes a series of yes/no decisions. Is it zero? Is
> it larger than one? And so on. The mantissa and sign are almost random, so they bypass the
> model and cost exactly one bit.
>
> Second, context: each decision picks a probability model based on neighbours already coded.
> If the left and top residuals were large, this one probably is too.
>
> Third, the adaptive binary arithmetic coder. This is the M-coder from H.264, with the same
> tables. It uses 64 probability states and no multiplication. There is no model in the header,
> because the encoder and decoder learn the same thing as they go. The header here is only
> 7 bytes.

**Show:** the learning chart, then the context table.

> Each context starts at 50 percent and learns quickly. It can never go below about 2 percent,
> which will matter in a moment. The binarization and contexts are my own, adapted to pixels,
> while the arithmetic engine follows the standard.

## 6. Results (5:00 to 6:40)

**Show:** Harnverhalt, window, MED, Results tab.

> Now the CT slice, cropped to 256 by 256. Huffman takes 4.557 bits per pixel, arithmetic
> 4.531 and CABAC 3.986. Three things stand out. Huffman and arithmetic are almost equal,
> because no symbol here needs less than one bit. Second, CABAC is 12 percent smaller. And
> third, it goes below the entropy of 4.49. That is not a contradiction, because that entropy
> only describes a model without context.

**Show:** Generated, Skewed binary, prediction None.

> Here is where arithmetic coding really matters. An image with 2 percent ones. The entropy is
> 0.14 bits. Huffman is stuck at 1 bit per symbol, while arithmetic coding gets 0.142: seven
> times smaller.

**Show:** Generated, Constant, then Uniform noise with MED.

> CABAC does not always win. On a constant image it costs 0.03 bits per pixel against 0.002
> for arithmetic coding, because of that probability floor. On pure noise there is nothing for
> the contexts to find, so it pays for learning: 9.08 against 8.87 bits. Prediction also hurts
> on noise, raising the entropy from 8 to 8.7 bits.

**Show:** Harnverhalt, stored 12 bit, MED, then the Benchmark tab.

> The stored 12-bit CT is the honest medical number. Here CABAC only gains 1 percent over
> arithmetic coding, 6.117 against 6.176, because the low bits are mostly noise. On the full
> 512 by 512 slice the ratio is about 2 to 1, for all three coders. For comparison, the DICOM
> file is 221 kilobytes with JPEG Lossless and metadata, and CABAC with MED gets 189.
>
> On speed, across the whole benchmark CABAC took about three times as long as arithmetic coding
> to encode and under twice as long to decode, and Huffman was fastest. These are Python
> implementations, so only the relative differences mean anything.

## 7. Conclusion (6:40 to 7:10)

**Show:** the Benchmark chart.

> To sum up. Prediction gives the largest gain on real images. Arithmetic coding only clearly
> beats Huffman when probabilities are very skewed. CABAC wins on real images because of
> context and needs no model in the header, but it costs the most time and loses on noise and
> near-constant data. So the right coder depends on the data. That is why CABAC is used in video,
> where residuals are skewed and structured, and speed is solved with dedicated hardware.
