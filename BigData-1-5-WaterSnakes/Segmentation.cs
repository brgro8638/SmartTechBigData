using OpenCvSharp;
using FellowOakDicom;
using FellowOakDicom.Imaging;

namespace BD_1_5_WaterSnakes
{
    public static class Segmentation
    {
        public class SegmentationParameters(
            int gaussianKSize, 
            int dilationKSize, 
            int morphKSize,
            int dilationIterations,
            double foregroundThreshold,
            bool useInvertedThreshold)
        {
            public readonly Size WatershedGaussianKernelSize = new(gaussianKSize, gaussianKSize);
            public readonly Size WatershedDilationKernelSize = new(dilationKSize, dilationKSize);
            public readonly Size SnakesMorphKernelSize = new(morphKSize, morphKSize);
            public readonly int WatershedDilationIterations = dilationIterations;
            public readonly double WatershedForegroundThreshold = foregroundThreshold;
            public readonly bool UseInvertedThreshold = useInvertedThreshold;
        }

        public static Mat? Load(string imagePath)
        {
            string imageType = imagePath.Split('.')[^1];
            Mat mat3;

            if (imageType == "dcm")
            {
                mat3 = LoadDicom(imagePath);
            }
            else if (new HashSet<string> { "jpg", "png", "jpeg", "bmp" }.Contains(imageType))
            {
                mat3 = LoadImage(imagePath);
            }
            else
            {
                Console.WriteLine("Can't load this type of image");
                return null;
            }

            return mat3;
        }

        private static Mat LoadImage(string path)
        {
            return Cv2.ImRead(path);
        }

        private static Mat LoadDicom(string path)
        {
            DicomFile dicomFile = DicomFile.Open(path);
            DicomImage dicomImage = new(dicomFile.Dataset);

            using IImage renderImage = dicomImage.RenderImage();
            byte[] pixels = renderImage.AsBytes();

            int width = renderImage.Width, height = renderImage.Height;

            using Mat mat4 = Mat.FromPixelData(height, width, MatType.CV_8UC4, pixels);

            // dropp alpha-kanalen
            Mat mat3 = new();
            Cv2.CvtColor(mat4, mat3, ColorConversionCodes.BGRA2BGR);

            return mat3;
        }

        public static Mat Watershed(Mat image, SegmentationParameters parameters)
        {
            // preprosesserer med gaussian blur for å begrense noise.
            using Mat matBlur = new();
            Cv2.GaussianBlur(image, matBlur, parameters.WatershedGaussianKernelSize, 0);

            // lager et grayscale-bilde som vi skal bruke til å genererer markers.
            using Mat gray = new();
            Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);

            // binær thresholding deler bildet i bakgrunn og forgrunn. otsu velger threshold automatisk.
            using Mat thresh = new();
            ThresholdTypes thresholdType = parameters.UseInvertedThreshold ? ThresholdTypes.BinaryInv : ThresholdTypes.Binary; // settes av bruker i GUI
            Cv2.Threshold(gray, thresh, 0, 255, thresholdType | ThresholdTypes.Otsu);

            // beregner avstanden fra hver pixel i forgrunnen til bakgrunnen, og normaliserer til intervalled [0, 1].
            using Mat dist = new();
            Cv2.DistanceTransform(thresh, dist, DistanceTypes.L2, DistanceTransformMasks.Mask3);
            Cv2.Normalize(dist, dist, 0, 1.0, NormTypes.MinMax);

            /*
            Cv2.ImShow("dist", dist);
            Cv2.WaitKey(0);
            */

            // bestemmer sikker forgrunn basert på normalisert avstand fra bakgrunn.
            using Mat sureFg = new();
            System.Diagnostics.Debug.WriteLine(parameters.WatershedForegroundThreshold);
            Cv2.Threshold(dist, sureFg, parameters.WatershedForegroundThreshold, 255, ThresholdTypes.Binary);

            sureFg.ConvertTo(sureFg, MatType.CV_8UC1);

            // utvider forgrunnen litt, og betegner alt ur, cotenfor som sikker bakgrunn. 
            using Mat sureBg = new();
            using Mat dilationKernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, parameters.WatershedDilationKernelSize);
            Cv2.Dilate(thresh, sureBg, dilationKernel, iterations: parameters.WatershedDilationIterations);

            // regioner som ikke er del av sikker forgrunn eller bakgrunn betegnes "unknown". disse er opp til watershed å klassifisere.
            using Mat unknown = new();
            Cv2.Subtract(sureBg, sureFg, unknown);

            // oppretter markers for å fortelle watershed hvor sikker forgrunn er.
            using Mat markers = new();
            Cv2.ConnectedComponents(sureFg, markers);
            Cv2.Add(markers, Scalar.All(1), markers);

            // pixler betegnet "unknown" settes til 0 (vi skalerte opp alt med 1 over, så bare unknown er 0).
            markers.SetTo(0, unknown);

            /*
            using Mat markersDisplay = new();
            Cv2.Normalize(markers, markersDisplay, 0, 255, NormTypes.MinMax, (int)MatType.CV_8UC1);
            Cv2.ImShow("markers", markersDisplay);
            Cv2.WaitKey(0);
            */

            // vi kjører watershed på det preprosesserte bildet, med markers.
            // grensene mellom forgrunn og bakgrunn blir -1
            Cv2.Watershed(matBlur, markers);

            // vi lager en maske hvor grensene mellom forgrunn og bakgrunn markeres med rødt.
            // deretter bruker vi masken til å fargelegge de aktuelle pixlene i det "originale" bgr bildet (mat3).
            Mat result = image.Clone();
            using Mat boundaryMask = new();
            Cv2.Compare(markers, new Scalar(-1), boundaryMask, CmpTypes.EQ);
            result.SetTo(new Scalar(0, 0, 255), boundaryMask);

            return result;
        }
        public static Mat BasicContours(Mat image, SegmentationParameters parameters)
        {
            Mat result = image.Clone();

            // grayscaling
            using Mat matGray = new();
            Cv2.CvtColor(result, matGray, ColorConversionCodes.BGRA2GRAY);

            // lager binært bilde ved å thresholde grayscale-bilde
            using Mat binary = new();
            ThresholdTypes thresholdType = parameters.UseInvertedThreshold ? ThresholdTypes.BinaryInv : ThresholdTypes.Binary; // settes av bruker i GUI
            Cv2.Threshold(matGray, binary, 0, 255, thresholdType | ThresholdTypes.Otsu);

            /*
            Cv2.ImShow("snakes thresholded", binary);
            Cv2.WaitKey(0);
            */

            // fjerner noise med morph. expansion
            using Mat morphKernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, parameters.SnakesMorphKernelSize);
            Cv2.MorphologyEx(binary, binary, MorphTypes.Close, morphKernel);

            /*
            Cv2.ImShow("snakes morphed", binary);
            Cv2.WaitKey(0);
            */

            // finner konturene i det binære bildet
            Cv2.FindContours(binary, out Point[][] contours, out HierarchyIndex[] hierarchy, RetrievalModes.Tree, ContourApproximationModes.ApproxSimple);

            // tegner konturene på det originale bildet (i grønt)
            for (int i = 0; i < contours.Length; i++)
            {
                Cv2.DrawContours(result, contours, i, Scalar.Green, 2);
            }

            return result;
        }
    }
}