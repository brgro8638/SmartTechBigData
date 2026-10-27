using Microsoft.Win32;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using OpenCvSharp;
using OpenCvSharp.WpfExtensions;

namespace ST_2_3_Pyramid
{
    public partial class MainWindow : System.Windows.Window 
    {
        private string? _selectedImagePath = null;
        private int _pyramidLayers = 4;
        private Mat _loadedImage = new();
        private List<Mat> _gaussianPyramid = [];
        private List<Mat> _laplacianPyramid = [];

        public MainWindow()
        {
            InitializeComponent();
        }

        private void SliderLevels_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            int newValue = (int)SliderLevels.Value;
            TxtLevelValue.Text = newValue.ToString();
            _pyramidLayers = newValue;
        }

        private void BtnLoadImage_Click(object sender, RoutedEventArgs e)
        {
            ResetUI();

            OpenFileDialog openFileDialog = new()
            {
                Title = "Select an Image",
                Filter = "Image Files (*.png;*.jpg;*.jpeg;*.bmp)|*.png;*.jpg;*.jpeg;*.bmp"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                _selectedImagePath = openFileDialog.FileName;
            }

            if (_selectedImagePath == null)
            {
                return;
            }

            _loadedImage = Cv2.ImRead(_selectedImagePath, ImreadModes.Color);
            ImgLoaded.Source = _loadedImage.ToBitmapSource();

            BtnBuildPyramids.IsEnabled = true;
        }

        private void BtnBuildPyramids_Click(object sender, RoutedEventArgs e)
        {
            _gaussianPyramid = Pyramids.BuildGaussianPyramid(_loadedImage, _pyramidLayers);
            _laplacianPyramid = Pyramids.BuildLaplacianPyramid(_gaussianPyramid);

            DisplayPyramidLayers(StackPanelGaussian, _gaussianPyramid);
            DisplayPyramidLayers(StackPanelLaplacian, _laplacianPyramid);

            TabPyramids.Visibility = Visibility.Visible;
            MainTabControl.SelectedItem = TabPyramids;
            BtnReconstructCompare.IsEnabled = true;
        }

        private void BtnReconstructCompare_Click(object sender, RoutedEventArgs e)
        {
            Mat reconstructedImage = Pyramids.GetReconstructedImage(_laplacianPyramid);

            ImgOriginal.Source = _loadedImage.ToBitmapSource();
            Mat displayFriendlyReconstruction = Pyramids.ToUINT8(reconstructedImage);
            ImgReconstructed.Source = displayFriendlyReconstruction.ToBitmapSource();

            (double, double) reconstructionMetrics = Pyramids.GetReconstructionError(_loadedImage, reconstructedImage);
            TxtReconstructInfo.Text = $"Original and Reconstructed Images - MSE: {reconstructionMetrics.Item1:E2}, Max Error: {reconstructionMetrics.Item2:E2}";

            TabReconstruct.Visibility = Visibility.Visible;
            MainTabControl.SelectedItem = TabReconstruct;
        }

        private void DisplayPyramidLayers(StackPanel targetPanel, List<Mat> pyramidImages)
        {
            targetPanel.Children.Clear();

            foreach (var mat in pyramidImages)
            {
                if (mat.Empty()) continue;

                Mat displayFriendlyMat = Pyramids.ToUINT8(mat);
                BitmapSource bitmapSource = displayFriendlyMat.ToBitmapSource();

                Image imgControl = new()
                {
                    Source = bitmapSource,
                    Stretch = Stretch.Uniform,
                    Width = 300,  
                    Height = 300,
                    Margin = new Thickness(5)
                };

                Border border = new()
                {
                    Background = Brushes.White,
                    BorderBrush = Brushes.LightGray,
                    BorderThickness = new Thickness(1),
                    Margin = new Thickness(0, 0, 0, 10), 
                    Child = imgControl
                };

                targetPanel.Children.Add(border);
            }
        }

        private void ResetUI()
        {
            TabPyramids.Visibility = Visibility.Collapsed;
            TabReconstruct.Visibility = Visibility.Collapsed;
            MainTabControl.SelectedItem = TabLoadedImage;
            BtnReconstructCompare.IsEnabled = false;
            BtnBuildPyramids.IsEnabled = false;
        }
    }
}