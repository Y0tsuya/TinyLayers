using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Text;
using System.Threading;
using TinyLayers;

//--------------------- API list ---------------------//
// enum ActivationType { Linear, ReLu, LeakyReLu, Sigmoid };
// enum LayerType { Base, Input, Dense, Output };
// enum BiasType { None, Shared, Unique };
// enum PointStyle { DOT, CROSS, EX };
//---------- Basic ----------//
// string[]? Args;
// void AddLogInfo(string text);
// void AddLogWarning(string text);
// void AddLogError(string text);
// void SetProgressIndicator(int value, int max);
//---------- Minnow-Specific ----------//
// void AddLog(string text);
// void WriteConsole(string text, Color textColor);
// void ClearConsole();
// void SetAsciiBoxXY(int x, int y);
// void WriteAsciiBoxXY(int x, int y, char c);
// void WriteAsciiBoxXY(int x, int y, string s);
// void ClearAsciiBox();
// void UpdateAsciiBox();
// void SetDataTableSize(int rows, int cols);
// void SetDataTableEntry(int row, int col, string text, Color fore, Color back);
// void SetBitmapBox1(Bitmap bmp);
// void SetBitmapBox2(Bitmap bmp);
// void UpdateBitmapBox1();
// void UpdateBitmapBox2();
// void ShowTab(int tabNum);
// void PlotCurve(Bitmap map, SampleS[] points, Color penColor, int penWidth,
//                RectangleF srcExtents, RectangleF dstExtents);
// void PlotPoints(Bitmap map, SampleS[] points, Color penColor, int penWidth,
//                PointStyle pointStyle, RectangleF srcExtents, RectangleF dstExtents);


SampleM[] XOR = new SampleM[4];
SampleM[] predict = new SampleM[4];
XOR[0] = new SampleM([0, 0], 0);
XOR[1] = new SampleM([0, 1], 1);
XOR[2] = new SampleM([1, 0], 1);
XOR[3] = new SampleM([1, 1], 0);
float mse;
Model model = new Model();

model.AddLayer(new InputLayer(inDims: 2));
model.AddLayer(new DenseLayer(outDims: 4, activation: ActivationType.ReLu, useBias: BiasType.Unique));
//model.AddLayer(new DenseLayer(outDims: 2, activation: ActivationType.ReLu, useBias: BiasType.Unique));
model.AddLayer(new OutputLayer(activation: ActivationType.Sigmoid));
BaseLayer.LearningRate = 0.5f;
BaseLayer.QAT = false;
int Epochs = 1000;

Model.CloneSet(XOR, predict);
for (int epoch = 0; epoch < Epochs; epoch++) {
	mse = model.TrainSet(XOR);
	model.PredictSet(predict);
	Console.WriteLine(String.Format("Epoch {0}, mse={1}", epoch, mse));
}
Console.WriteLine("Input | Output");
Console.WriteLine(" 0 0  | " + predict[0].Y);
Console.WriteLine(" 0 1  | " + predict[1].Y);
Console.WriteLine(" 1 0  | " + predict[2].Y);
Console.WriteLine(" 1 1  | " + predict[3].Y);
string weights = model.GetLayerWeights();
