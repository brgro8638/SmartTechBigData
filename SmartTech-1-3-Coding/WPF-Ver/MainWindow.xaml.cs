using Microsoft.Win32;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;

namespace ST_1_3_Coding
{
    public partial class MainWindow : Window
    {
        private string _selectedFilePath = string.Empty;
        private bool _formatHuffman = false;
        private bool _formatArithmetic = false;

        public MainWindow()
        {
            InitializeComponent();

            // kobler opp event listeners
            BtnSelectFile.Click += BtnSelectFile_Click;
            BtnRunCompression.Click += BtnRunCompression_Click;

            BtnRunCompression.IsEnabled = false;
        }

        #region Button Handlers

        private void BtnSelectFile_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new()
            {
                Title = "Select File for Compression Benchmark",
                Filter = "All Files (*.*)|*.*"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                _selectedFilePath = openFileDialog.FileName;
                TxtFilePath.Text = $"Selected: {Path.GetFileName(_selectedFilePath)}";
                TxtFilePath.Foreground = System.Windows.Media.Brushes.Black;
            }

            BtnRunCompression.IsEnabled = true;
            ResetTestResults();
        }

        private async void BtnRunCompression_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_selectedFilePath) || !File.Exists(_selectedFilePath))
            {
                MessageBox.Show("Please select a valid file before running compression.",
                                "No File Selected", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            BtnRunCompression.IsEnabled = false;

            try
            {
                byte[] fileBytes = await File.ReadAllBytesAsync(_selectedFilePath);

                // kjører én komprimering om gangen for å minimere deling av ressurser (pga tidsmåling)
                RunRleCompression(fileBytes);
                RunHuffmanCompression(fileBytes);
                RunArithmeticCompression(fileBytes);
                RunCabacCompression(fileBytes);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"An error occurred during compression: {ex.Message}",
                                "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                BtnRunCompression.IsEnabled = true;
            }
        }

        private void CheckBox_Toggled(object sender, RoutedEventArgs e)
        {
            if (sender is CheckBox checkBox)
            {
                switch (checkBox.Name)
                {
                    case "CheckHuffman":
                        _formatHuffman = checkBox.IsChecked == true;
                        break;
                    case "CheckArithmetic":
                        _formatArithmetic= checkBox.IsChecked == true;
                        break;
                }
            }
        }

        private void ResetTestResults()
        {
            string speedPlaceholder = "0 MB/s";
            string ratioPlaceholder = "0:0 (0%)";
            string roundTripPlaceholder = "N/A";

            foreach (var e in new[] { TxtRleSpeed, TxtHuffmanSpeed, TxtArithmeticSpeed, TxtCabacSpeed })
            {
                e.Text = speedPlaceholder;
            }

            foreach (var e in new[] { TxtRleRatio, TxtHuffmanRatio, TxtArithmeticRatio, TxtCabacRatio })
            {
                e.Text = ratioPlaceholder;
            }

            foreach (var e in new[] { TxtRleRT, TxtHuffmanRT, TxtArithmeticRT, TxtCabacRT })
            {
                e.Text = roundTripPlaceholder;
            }
        }

        #endregion
        #region Compression Algorithms

        private void RunRleCompression(byte[] fileBytes)
        {
            Stopwatch sw = Stopwatch.StartNew();
            byte[] encodedBytes = RLE.Encode(fileBytes);
            double elapsed = sw.Elapsed.TotalMilliseconds;

            TxtRleSpeed.Text = GetCompressionSpeed(fileBytes.Length, elapsed);
            TxtRleRatio.Text = GetCompressionRatio(fileBytes.Length, encodedBytes.Length);

            byte[] decodedBytes = RLE.Decode(encodedBytes);
            TxtRleRT.Text = GetRoundTripPassed(fileBytes, decodedBytes);
            TxtRlePreview.Text = GetEncodedPreview(encodedBytes);
        }

        private void RunHuffmanCompression(byte[] fileBytes)
        {
            Stopwatch sw = Stopwatch.StartNew();
            (byte[] encodedBytes, int bitLength, Huffman.Node? root) = Huffman.Encode(fileBytes);
            double elapsed = sw.Elapsed.TotalMilliseconds;

            TxtHuffmanSpeed.Text = GetCompressionSpeed(fileBytes.Length, elapsed);
            TxtHuffmanRatio.Text = GetCompressionRatio(fileBytes.Length, encodedBytes.Length);

            byte[] decodedBytes = Huffman.Decode(encodedBytes, bitLength, root);
            TxtHuffmanRT.Text = GetRoundTripPassed(fileBytes, decodedBytes);
            PreviewFormat? format = _formatHuffman ? PreviewFormat.BitStream : null;
            TxtHuffmanPreview.Text = GetEncodedPreview(encodedBytes, format);
        }

        private void RunArithmeticCompression(byte[] fileBytes)
        {
            Stopwatch sw = Stopwatch.StartNew();
            byte[] encodedBytes = Arithmetic.Encode(fileBytes);
            double elapsed = sw.Elapsed.TotalMilliseconds;

            TxtArithmeticSpeed.Text = GetCompressionSpeed(fileBytes.Length, elapsed);
            TxtArithmeticRatio.Text = GetCompressionRatio(fileBytes.Length, encodedBytes.Length);

            byte[] decodedBytes = Arithmetic.Decode(encodedBytes);
            TxtArithmeticRT.Text = GetRoundTripPassed(fileBytes, decodedBytes);
            PreviewFormat? format = _formatArithmetic ? PreviewFormat.Float : null;
            TxtArithmeticPreview.Text = GetEncodedPreview(encodedBytes, format);
        }

        private void RunCabacCompression(byte[] fileBytes)
        {
            Stopwatch sw = Stopwatch.StartNew();
            byte[] encodedBytes = Cabac.Encode(fileBytes);
            double elapsed = sw.Elapsed.TotalMilliseconds;

            TxtCabacSpeed.Text = GetCompressionSpeed(fileBytes.Length, elapsed);
            TxtCabacRatio.Text = GetCompressionRatio(fileBytes.Length, encodedBytes.Length);

            byte[] decodedBytes = Cabac.Decode(encodedBytes);
            TxtCabacRT.Text = GetRoundTripPassed(fileBytes, decodedBytes);
            TxtCabacPreview.Text = GetEncodedPreview(encodedBytes);
        }

        #endregion
        #region Compression Stat Helpers

        private string GetCompressionSpeed(int bytes, double elapsedMilliseconds)
        {
            return $"{Math.Round((bytes / 1000.0) / elapsedMilliseconds, 2)} MB/s";
        }

        private string GetCompressionRatio(int oldBytes, int newBytes)
        {
            double ratio = (double)oldBytes / newBytes;
            double percentageReduction = (1 - (1 / ratio)) * 100;
            return $"{Math.Round(ratio, 2)}:1 ({Math.Round(percentageReduction, 2)}% red.)";
        }

        private string GetRoundTripPassed(byte[] pre, byte[] post)
        {
            return pre.SequenceEqual(post).ToString();
        }

        enum PreviewFormat { Float, BitStream } // alternative måter å illustrere output fra algoritmene (alternativer til rene bytes)

        private string GetEncodedPreview(byte[] encoded, PreviewFormat? format = null)
        {
            return format switch
            {
                PreviewFormat.Float => GetFloatFormat(encoded) + "...",
                PreviewFormat.BitStream => string.Join("", encoded[0..8].Select(b => Convert.ToString(b, 2).PadLeft(8, '0'))) + "...",
                _ => string.Join(',', encoded[0..32]) + "...",
            };
        }

        private string GetFloatFormat(byte[] encoded)
        {
            ReadOnlySpan<byte> slice = encoded.AsSpan(0, 16);

            double value = 0.0;
            double weight = 0.5; // Starts at 2^-1

            foreach (byte b in slice)
            {
                for (int bit = 7; bit >= 0; bit--)
                {
                    if (((b >> bit) & 1) == 1)
                    {
                        value += weight;
                    }
                    weight /= 2.0;
                }
            }

            return value.ToString("G17");
        }

        #endregion
    }
}
