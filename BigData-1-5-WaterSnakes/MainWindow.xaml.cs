using Microsoft.Win32;
using System.Windows;
using OpenCvSharp;
using OpenCvSharp.WpfExtensions;
using System.Windows.Controls;

namespace BD_1_5_WaterSnakes
{
    public partial class MainWindow : System.Windows.Window
    {
        private string _selectedImagePath = string.Empty;
        private bool _useInvertedThreshold = false;
        private Mat _loadedImage = new();

        public MainWindow()
        {
            InitializeComponent();
        }

        private void BtnLoadImage_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new()
            {
                Title = "Select an Image",
                Filter = "Image Files (*.png;*.jpg;*.jpeg;*.bmp;*.dcm)|*.png;*.jpg;*.jpeg;*.bmp;*.dcm"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                _selectedImagePath = openFileDialog.FileName;
                txtLoadedFile.Text = _selectedImagePath;
            } 
            else
            {
                return;
            }

            Mat? mat3 = Segmentation.Load(_selectedImagePath);

            if (mat3 == null)
            {
                MessageBox.Show("Failed to load the image. Please check the file format.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            } 
            else
            {
                _loadedImage?.Dispose();
                _loadedImage = mat3;
            }

            tabControl.SelectedItem = tabOriginals;
            tabResults.Visibility = Visibility.Collapsed;

            imgOriginalLeft.Source = mat3.ToBitmapSource();
            imgOriginalRight.Source = mat3.ToBitmapSource();
        }
        private void BtnRunSegmentation_Click(object sender, RoutedEventArgs e)
        {
            Segmentation.SegmentationParameters parameters = new(
                (int)sliderWatershedGaussianKernel.Value,
                (int)sliderWatershedDilationKernel.Value,
                (int)sliderSnakesMorphKernel.Value,
                (int)sliderWatershedDilationIterations.Value,
                (double)sliderWatershedForegroundThreshold.Value,
                _useInvertedThreshold
            );

            Mat watershedResult = Segmentation.Watershed(_loadedImage, parameters);
            Mat snakesResult = Segmentation.BasicContours(_loadedImage, parameters);

            imgSegmentedLeft.Source = watershedResult.ToBitmapSource();
            imgSegmentedRight.Source = snakesResult.ToBitmapSource();

            tabResults.Visibility = Visibility.Visible;
            tabControl.SelectedItem = tabResults;
        }

        private void ChkInvertedThreshold_Toggled(object sender, RoutedEventArgs e)
        {
            if (sender is CheckBox checkBox)
            {
                _useInvertedThreshold = checkBox.IsChecked ?? false;
            }
        }
    }
}