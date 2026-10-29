using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace ST_1_6_Pipeline
{
    internal static class TreeViewer
    {
        private class NodePosition
        {
            public Huffman.Node? Node { get; set; }
            public double X { get; set; }
            public double Y { get; set; }
        }

        private static List<NodePosition> CalculateLayout(Huffman.Node root, double x, double y, double hSpacing, double vSpacing)
        {
            List<NodePosition> positions = new List<NodePosition>();
            if (root == null) return positions;

            // Add current node
            positions.Add(new NodePosition { Node = root, X = x, Y = y });

            // Recursively calculate children with reduced horizontal spacing
            if (root.Left != null)
            {
                positions.AddRange(CalculateLayout(root.Left, x - hSpacing, y + vSpacing, hSpacing / 2, vSpacing));
            }
            if (root.Right != null)
            {
                positions.AddRange(CalculateLayout(root.Right, x + hSpacing, y + vSpacing, hSpacing / 2, vSpacing));
            }

            return positions;
        }

        public static void DrawTree(Canvas canvas, Huffman.Node root)
        {
            canvas.Children.Clear();
            canvas.Background = Brushes.White;
            EnablePanAndZoom(canvas);

            List<NodePosition> positions = CalculateLayout(root, 400, 30, 200, 60);
            Dictionary<Huffman.Node, NodePosition> posDict = positions.ToDictionary(p => p.Node ?? new Huffman.Node(0, 0), p => p);

            // Tegn linjer (kanter)
            foreach (NodePosition pos in positions)
            {
                if (pos.Node?.Left != null && posDict.ContainsKey(pos.Node.Left))
                {
                    NodePosition leftPos = posDict[pos.Node.Left];

                    canvas.Children.Add(new Line
                    {
                        X1 = pos.X,
                        Y1 = pos.Y,
                        X2 = leftPos.X,
                        Y2 = leftPos.Y,
                        Stroke = Brushes.Gray,
                        StrokeThickness = 2
                    });

                    AddEdgeLabel(canvas, (pos.X + leftPos.X) / 2, (pos.Y + leftPos.Y) / 2, "0");
                }

                if (pos.Node?.Right != null && posDict.ContainsKey(pos.Node.Right))
                {
                    NodePosition rightPos = posDict[pos.Node.Right];

                    canvas.Children.Add(new Line
                    {
                        X1 = pos.X,
                        Y1 = pos.Y,
                        X2 = rightPos.X,
                        Y2 = rightPos.Y,
                        Stroke = Brushes.Gray,
                        StrokeThickness = 2
                    });

                    AddEdgeLabel(canvas, (pos.X + rightPos.X) / 2, (pos.Y + rightPos.Y) / 2, "1");
                }
            }

            // Tegn noder
            foreach (NodePosition pos in positions)
            {
                Grid grid = new() { Width = 40, Height = 40 };

                grid.Children.Add(new Ellipse
                {
                    Fill = Brushes.LightSteelBlue,
                    Stroke = Brushes.DarkSlateBlue,
                    StrokeThickness = 2
                });

                grid.Children.Add(new TextBlock
                {
                    Text = $"{Convert.ToChar(pos.Node?.Byte ?? 0)}\n{pos.Node?.Freq}",
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Center,
                    TextAlignment = TextAlignment.Center,
                    FontSize = 9
                });

                Canvas.SetLeft(grid, pos.X - 20);
                Canvas.SetTop(grid, pos.Y - 20);

                canvas.Children.Add(grid);
            }
        }

        private static void AddEdgeLabel(Canvas canvas, double x, double y, string labelText)
        {
            Border border = new()
            {
                Background = Brushes.White,
                Padding = new Thickness(2, 0, 2, 0),
                Child = new TextBlock
                {
                    Text = labelText,
                    FontSize = 10,
                    FontWeight = FontWeights.Bold,
                    Foreground = Brushes.DarkSlateGray
                }
            };

            Canvas.SetLeft(border, x - 8);
            Canvas.SetTop(border, y - 8);

            canvas.Children.Add(border);
        }

        private static void EnablePanAndZoom(Canvas canvas)
        {
            if (canvas.Tag is bool isInitialized && isInitialized)
            {
                return;
            }

            canvas.Tag = true;

            TransformGroup transformGroup = new();
            ScaleTransform scaleTransform = new(1, 1);
            TranslateTransform translateTransform = new(0, 0);

            transformGroup.Children.Add(scaleTransform);
            transformGroup.Children.Add(translateTransform);
            canvas.RenderTransform = transformGroup;
            canvas.RenderTransformOrigin = new Point(0, 0);

            bool isDragging = false;
            Point startPoint = new();

            canvas.MouseLeftButtonDown += (s, e) =>
            {
                isDragging = true;
                startPoint = e.GetPosition(canvas.Parent as UIElement ?? canvas);
                canvas.CaptureMouse();
            };

            canvas.MouseLeftButtonUp += (s, e) =>
            {
                isDragging = false;
                canvas.ReleaseMouseCapture();
            };

            canvas.MouseMove += (s, e) =>
            {
                if (isDragging)
                {
                    UIElement? parent = canvas.Parent as UIElement;
                    Point currentPoint = e.GetPosition(parent ?? canvas);
                    Vector delta = currentPoint - startPoint;

                    translateTransform.X += delta.X;
                    translateTransform.Y += delta.Y;

                    startPoint = currentPoint;
                }
            };

            canvas.PreviewMouseWheel += (s, e) =>
            {
                e.Handled = true;
                UIElement? parent = canvas.Parent as UIElement;
                Point mousePos = e.GetPosition(parent ?? canvas);

                double zoom = e.Delta > 0 ? 1.15 : 1 / 1.15;
                double newScale = scaleTransform.ScaleX * zoom;

                if (newScale < 0.1 || newScale > 10.0) return;

                scaleTransform.ScaleX = newScale;
                scaleTransform.ScaleY = newScale;

                translateTransform.X = mousePos.X - (mousePos.X - translateTransform.X) * zoom;
                translateTransform.Y = mousePos.Y - (mousePos.Y - translateTransform.Y) * zoom;
            };
        }
    }
}