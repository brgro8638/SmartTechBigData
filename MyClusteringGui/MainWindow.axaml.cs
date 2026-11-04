using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Media.Imaging;
using Avalonia.Platform.Storage;

namespace MyClusteringGui;

public partial class MainWindow : Window
{
    private string? selectedZipPath;

    public MainWindow()
    {
        InitializeComponent();
    }

    private async void ChooseZipButton_OnClick(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var storageProvider = StorageProvider;
        var files = await storageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Choose Harnverhalt2.zip",
            AllowMultiple = false,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("ZIP archive") { Patterns = new[] { "*.zip" } }
            }
        });

        if (files.Count == 0)
        {
            return;
        }

        selectedZipPath = files[0].Path.LocalPath;
        StatusText.Text = $"Selected: {selectedZipPath}";
        LoadImageButton.IsEnabled = true;
    }

    private async void LoadImageButton_OnClick(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (selectedZipPath is null)
        {
            return;
        }

        ChooseZipButton.IsEnabled = false;
        LoadImageButton.IsEnabled = false;
        StatusText.Text = "Reading DICOM and processing with OpenCV...";

        try
        {
            var previewPath = Path.Combine(Path.GetTempPath(), $"harnverhalt-{Guid.NewGuid():N}.png");
            var result = await RunPythonLoaderAsync(selectedZipPath, previewPath);
            if (result.ExitCode != 0)
            {
                throw new InvalidOperationException(result.Error.Trim());
            }

            PreviewImage.Source = new Bitmap(previewPath);
            StatusText.Text = result.Output.Trim();
        }
        catch (Exception exception)
        {
            StatusText.Text = $"Could not load image: {exception.Message}";
        }
        finally
        {
            ChooseZipButton.IsEnabled = true;
            LoadImageButton.IsEnabled = true;
        }
    }

    private static async Task<(int ExitCode, string Output, string Error)> RunPythonLoaderAsync(
        string zipPath,
        string previewPath)
    {
        var scriptPath = Path.Combine(AppContext.BaseDirectory, "harnverhalt.py");
        var python = Environment.GetEnvironmentVariable("PYTHON_EXECUTABLE");
        if (string.IsNullOrWhiteSpace(python))
        {
            var condaPrefix = Environment.GetEnvironmentVariable("CONDA_PREFIX");
            python = string.IsNullOrWhiteSpace(condaPrefix)
                ? "python3"
                : Path.Combine(condaPrefix, "bin", "python");
        }
        var startInfo = new ProcessStartInfo
        {
            FileName = python,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        startInfo.ArgumentList.Add(scriptPath);
        startInfo.ArgumentList.Add(zipPath);
        startInfo.ArgumentList.Add("--output");
        startInfo.ArgumentList.Add(previewPath);

        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Could not start Python. Set PYTHON_EXECUTABLE to your bigdata Python path.");
        var outputTask = process.StandardOutput.ReadToEndAsync();
        var errorTask = process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();
        return (process.ExitCode, await outputTask, await errorTask);
    }
}