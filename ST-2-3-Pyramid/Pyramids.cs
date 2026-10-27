using OpenCvSharp;

namespace ST_2_3_Pyramid
{
    static class Pyramids
    {
        public static List<Mat> BuildGaussianPyramid(Mat image, int pyramidLayers)
        {
            if (image == null || image.Empty())
                throw new ArgumentException("Image is empty.");

            Mat floatImage = new();
            image.ConvertTo(floatImage, MatType.CV_32FC(image.Channels()));

            List<Mat> images = [floatImage];
            Mat current = images[0];

            for (int i = 0; i < pyramidLayers; i++)
            {
                if (current.Rows < 2 || current.Cols < 2)
                    break;

                Mat next = new();
                Cv2.PyrDown(current, next);

                if (next.Empty())
                    break;

                images.Add(next);
                current = next;
            }

            return images;
        }

        public static List<Mat> BuildLaplacianPyramid(List<Mat> gaussianPyramid)
        {
            if (gaussianPyramid.Count < 2)
            {
                throw new ArgumentException("Insufficient levels in gaussian pyramid");
            }

            List<Mat> images = [];

            for (int i = 0; i < gaussianPyramid.Count - 1; i++)
            {
                Mat current = gaussianPyramid[i], next = gaussianPyramid[i + 1];
                using Mat upscaled = new();
                Cv2.PyrUp(next, upscaled, new Size(current.Cols, current.Rows));

                Mat difference = new();
                Cv2.Subtract(current, upscaled, difference);
                images.Add(difference);
            }

            images.Add(gaussianPyramid[^1].Clone());
            return images;
        }

        public static Mat GetReconstructedImage(List<Mat> laplacianPyramid)
        {
            if (laplacianPyramid.Count < 2)
            {
                throw new ArgumentException("Insufficient levels in laplacian pyramid");
            }

            Mat recBase = laplacianPyramid[^1].Clone(); // basis for rekonstruksjon

            for (int i = laplacianPyramid.Count - 2; i > -1; i--)
            {
                Mat nextLevel = laplacianPyramid[i];
                Cv2.PyrUp(recBase, recBase, new Size(nextLevel.Cols, nextLevel.Rows));
                Cv2.Add(recBase, nextLevel, recBase);
            }

            return recBase;
        }

        public static Mat ToUINT8(Mat floatImage)
        {
            Mat uint8Image = new();
            floatImage.ConvertTo(uint8Image, MatType.CV_8UC3);
            return uint8Image;
        }

        public static (double meanSquareError, double maxError) GetReconstructionError(Mat originalImage, Mat reconstructedImage)
        {
            // Original -> Float32
            Mat floatImage = new();
            originalImage.ConvertTo(floatImage, MatType.CV_32FC(originalImage.Channels()));

            Mat differenceImage = new();
            Cv2.Subtract(floatImage, reconstructedImage, differenceImage);

            double meanSquareError =
                Cv2.Norm(differenceImage, NormTypes.L2SQR) /
                (differenceImage.Rows * differenceImage.Cols * differenceImage.Channels());

            double maxError = Cv2.Norm(differenceImage, NormTypes.INF);

            return (meanSquareError, maxError);
        }
    }
}
