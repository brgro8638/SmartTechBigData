using Microsoft.Win32;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.IO;
using System.Diagnostics;

namespace ST_1_6_Pipeline
{
    public partial class MainWindow : Window
    {
        private string _selectedFilePath = string.Empty;
        private Huffman.Node? _root = null;
        private Dictionary<byte, string>? _codebook = null;
        private byte[]? _compressedBytes = null; // I tilfelle vi ønsker å kunne eksportere eller bruke de komprimerte dataene senere
        private bool _roundTripCheckEnabled = true;

        public MainWindow()
        {
            InitializeComponent();

            BtnSelectFile.Click += SelectFileButton_Click;
            BtnRunCompression.Click += RunCompressionButton_Click;
            BtnVisualizeTree.Click += VisualizeTreeButton_Click;
            BtnVisualizeCodebook.Click += VisualizeCodebookButton_Click;
            ChkConfirmRoundTrip.Click += RoundTripCheck_Click;

            BtnRunCompression.IsEnabled = false;
            BtnVisualizeTree.IsEnabled = false;
            BtnVisualizeCodebook.IsEnabled = false;
        }

        private void SelectFileButton_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new()
            {
                Title = "Select File for Compression Benchmark",
                Filter = "All Files (*.*)|*.*"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                _selectedFilePath = openFileDialog.FileName;
                TxtFilePath.Text = $"Selected: {System.IO.Path.GetFileName(_selectedFilePath)}";
                TxtFilePath.Foreground = Brushes.Black;
            }

            BtnRunCompression.IsEnabled = true;
        }

        #region Visualization

        private void VisualizeTreeButton_Click(object sender, RoutedEventArgs e)
        {
            Window window = new()
            {
                Title = "Huffman Tree Visualization",
                Width = 800,
                Height = 600
            };

            Canvas canvas = new()
            {
                Background = Brushes.White
            };

            ScrollViewer scrollViewer = new()
            {
                HorizontalScrollBarVisibility = ScrollBarVisibility.Auto,
                VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
                Content = canvas
            };

            window.Content = scrollViewer;
            window.Show();

            if (_root != null)
            {
                TreeViewer.DrawTree(canvas, _root);
            }
        }

        public void VisualizeCodebookButton_Click(object sender, RoutedEventArgs e)
        {
            Window window = new()
            {
                Title = "Dictionary Viewer",
                Width = 250,
                Height = 400,
                WindowStartupLocation = WindowStartupLocation.CenterScreen
            };

            List<string> displayList = _codebook?
                .OrderBy(kvp => kvp.Value.Length)
                .Select(kvp => $"{kvp.Key} / {Convert.ToChar(kvp.Key)} : {kvp.Value}")
                .ToList() ?? ["The codebook is empty."];

            ListBox listBox = new()
            {
                Margin = new Thickness(10),
                ItemsSource = displayList
            };

            window.Content = listBox;
            window.Show();
        }

        #endregion

        private void RunCompressionButton_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_selectedFilePath))
            {
                MessageBox.Show("Please select a file first.");
                return;
            }

            byte[] fileBytes = File.ReadAllBytes(_selectedFilePath);
            TxtPipeOriginal.Text = $"{fileBytes.Length} bytes ({ByteToMegabyteString(fileBytes.Length)})";
            TxtRleOriginal.Text = $"{fileBytes.Length} bytes ({ByteToMegabyteString(fileBytes.Length)})";

            // Huffman
            Stopwatch stopwatch = new();
            stopwatch.Start();
            _root = Huffman.BuildHuffmanTree(fileBytes);
            // _root = Huffman.BuildHuffmanTree(frequencyTable); // overload for eksisterende frequency table
            _codebook = Huffman.BuildHuffmanCodebook(_root);
            Huffman.Result huffmanResult = Huffman.Encode(fileBytes, _root, _codebook); // overload for eksisterende tree og codebook
            int _bitLength = huffmanResult.bitLength;

            // CABAC
            _compressedBytes = Cabac.Encode(huffmanResult.encoded);
            stopwatch.Stop();
            double pipelineMilliseconds = stopwatch.Elapsed.TotalMilliseconds;

            // Pipeline Stats
            TxtPipeEncoded.Text = $"{_compressedBytes.Length} bytes ({ByteToMegabyteString(_compressedBytes.Length)})";
            TxtPipeRatio.Text = GetCompressionRatio(fileBytes.Length, _compressedBytes.Length);
            TxtPipeSpeed.Text = GetCompressionSpeed(fileBytes.Length, pipelineMilliseconds);

            // RLE
            stopwatch.Restart();
            byte[] _encoded = NaiveRLE.Encode(fileBytes);
            stopwatch.Stop();
            double comparisonMilliseconds = stopwatch.Elapsed.TotalMilliseconds;

            // RLE Stats
            TxtRleEncoded.Text = $"{_encoded.Length} bytes ({ByteToMegabyteString(_encoded.Length)})";
            TxtRleRatio.Text = GetCompressionRatio(fileBytes.Length, _encoded.Length);
            TxtRleSpeed.Text = GetCompressionSpeed(fileBytes.Length, comparisonMilliseconds);

            if (_roundTripCheckEnabled)
            {
                // Pipeline
                byte[] _intermediateBytes = Cabac.Decode(_compressedBytes);
                byte[] _reconstructedBytes = Huffman.Decompress(_intermediateBytes, _bitLength, _root);
                TxtPipeRoundTrip.Text = GetRoundTripPassed(fileBytes, _reconstructedBytes);

                // RLE
                byte[] _decoded = NaiveRLE.Decode(_encoded);
                TxtRleRoundTrip.Text = GetRoundTripPassed(fileBytes, _decoded);
            }
            else
            {
                TxtPipeRoundTrip.Text = "N/A";
                TxtRleRoundTrip.Text = "N/A";
            }

            BtnVisualizeTree.IsEnabled = true;
            BtnVisualizeCodebook.IsEnabled = true;
        }

        private void RoundTripCheck_Click(object sender, RoutedEventArgs e)
        {
            if (sender is CheckBox checkBox)
            {
                _roundTripCheckEnabled = checkBox.IsChecked == true;
            }
        }

        #region Compression Stat Helpers

        private string GetCompressionSpeed(int oldbytes, double elapsedMilliseconds)
        {
            return $"{Math.Round((oldbytes / 1000.0) / elapsedMilliseconds, 2)} MB/s";
        }

        private string GetCompressionRatio(int oldBytes, int newBytes)
        {
            double ratio = (double)oldBytes / newBytes;
            double percentageReduction = (1 - (1 / ratio)) * 100;
            return $"{Math.Round(ratio, 2)}:1 ({Math.Round(percentageReduction, 2)}% red.)";
        }

        private string GetRoundTripPassed(byte[] oldBytes, byte[] newBytes)
        {
            return oldBytes.SequenceEqual(newBytes).ToString();
        }

        private string ByteToMegabyteString(int bytes)
        {
            return $"{Math.Round(bytes / 1000000.0, 2)} MB";
        }

        #endregion
    }
}