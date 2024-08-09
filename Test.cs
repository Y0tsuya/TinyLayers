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
