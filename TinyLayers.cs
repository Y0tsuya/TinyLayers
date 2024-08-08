using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

namespace TinyLayers {

	public enum ActivationType { Linear, ReLu, LeakyReLu, Sigmoid, Tanh };

	public enum LayerType { Base, Input, Dense, Output };

	public enum BiasType { None, Shared, Unique };

	public class Activation {
		public static int Prec = 24;	// fixed-point decimal precision
		public static int[] SigmoidLut;
		public static int[] TanhLut;


		public static int Activate(ActivationType type, int x) {
			switch (type) {
				case ActivationType.Linear: return Clamp(x, -128, 127);
				case ActivationType.ReLu: return ReLu(x);
				case ActivationType.LeakyReLu: return LeakyReLu(x);
				case ActivationType.Sigmoid: return Sigmoid(x);
				case ActivationType.Tanh: return Tanh(x);
				default: return x;
			}
		}

		public static float Activate(ActivationType type, float x) {
			switch (type) {
				case ActivationType.Linear: return Clamp(x, -1, 1);
				case ActivationType.ReLu: return ReLu(x);
				case ActivationType.LeakyReLu: return LeakyReLu(x);
				case ActivationType.Sigmoid: return Sigmoid(x);
				case ActivationType.Tanh: return Tanh(x);
				default: return x;
			}
		}

		public static float Gradient(ActivationType type, float x) {
			switch (type) {
				case ActivationType.Linear: return 1;
				case ActivationType.ReLu: return ReLuDx(x);
				case ActivationType.LeakyReLu: return LeakyReLuDx(x);
				case ActivationType.Sigmoid: return SigmoidDx(x);
				case ActivationType.Tanh: return TanhDx(x);
				default: return x; 
			}
		}

		public static float Clamp(float x, float min, float max) {
			if (x < min) return min;
			else if (x > max) return max;
			else return x;
		}

		public static int Clamp(int x, int min, int max) {
			if (x < min) return min;
			else if (x > max) return max;
			else return x;
		}

		// ReLu float
		public static float ReLu(float x) {
			if (x > 0) return x;
			else return 0;
		}

		public static float ReLuDx(float x) {
			if (x > 0) return 1;
			else return 0;
		}

		// ReLu int
		public static int ReLu(int x) {
			if (x > 0) return x;
			else return 0;
		}

		// LeakyReLu float
		public static float LeakyReLu(float x) {
			if (x > 0) return x;
			else return x * 0.125f;
		}

		public static float LeakyReLuDx(float x) {
			if (x > 0) return 1;
			else return 0.125f;
		}

		// LeakyReLu int
		public static int LeakyReLu(int x) {
			if (x > 0) return x;
			else return x / 8;
		}

		// Sigmoid float
		public static float Sigmoid(float x) {
			return (float)(1f / (1f + Math.Exp(-x)));
		}

		public static float SigmoidDx(float x) {
			return x * (1 - x);
		}

		public static int Sigmoid(int x) {
			bool isNegative = false;
			if (x < 0) {
				isNegative = true;
				x = -x;
			}
			byte lutSelect = (byte)(x >> Prec);
			int index;
			if (lutSelect <= 0x01) index = (x >> Prec - 6);
			else if (lutSelect <= 0x05) index = (x >> Prec - 4) + 96;
			else index = (x >> Prec - 2) + 168;
			if (index > 255) index = 255;
			int folded = SigmoidLut[index];
			if (isNegative) folded = -folded;
			folded = folded + 128;
			int scaled = folded << (Prec - 8);   // scale back to 16.16 fixed-poing
			return scaled;
		}

		// Tanh float
		public static float Tanh(float x) {
			//float exp = (float)Math.Exp(2 * x);
			//return (exp - 1) / (exp + 1);
			return (float)(2f / (1f + Math.Exp(-2 * x))) - 1f;
		}

		public static float TanhDx(float x) {
			return 1 - (x * x);
		}

		// Tanh int
		public static int Tanh(int x) {
			bool isNegative = false;
			if (x < 0) {
				isNegative = true;
				x = -x;
			}
			byte lutSelect = (byte)(x >> Prec);
			int index;
			if (lutSelect <= 0x01) index = (x >> Prec - 6);
			else if (lutSelect <= 0x05) index = (x >> Prec - 5) + 64;
			else index = (x >> Prec - 2) + 176;
			if (index > 255) index = 255;
			int folded = TanhLut[index];
			if (isNegative) folded = -folded;
			int scaled = folded << (Prec - 8);   // scale back to 16.16 fixed-poing
			return scaled;
		}

		public static void CreateSigmoidLut() {
			SigmoidLut = new int[256];
			float x, y;
			int i;
			for (i = 0; i < 128; i++) {
				x = (float)i / 64f;
				y = Sigmoid(x) * 255.5f;
				SigmoidLut[i] = (int)Math.Round(y, 0) - 128;
			}
			for (i = 128; i < 192; i++) {
				x = (float)i / 16f;
				y = Sigmoid(x) * 255.5f;
				SigmoidLut[i] = (int)Math.Round(y, 0) - 128;
			}
			for (i = 192; i < 256; i++) {
				x = (float)i / 4f;
				y = Sigmoid(x) * 255.5f;
				SigmoidLut[i] = (int)Math.Round(y, 0) - 128;
			}
		}

		public static void CreateTanhLut() {
			TanhLut = new int[256];
			float x, y;
			int i;
			for (i = 0; i < 128; i++) {
				x = (float)i / 128f;
				y = Tanh(x) * 127f;
				TanhLut[i] = (int)Math.Round(y, 0);
			}
			for (i = 128; i < 192; i++) {
				x = (float)i / 64f;
				y = Tanh(x) * 127f;
				TanhLut[i] = (int)Math.Round(y, 0);
			}
			for (i = 192; i < 256; i++) {
				x = (float)i / 8f;
				y = Tanh(x) * 127f;
				TanhLut[i] = (int)Math.Round(y, 0);
			}
		}
	}

	public class Matrix {
		public static float Dot(float[] a, float[] b) {
			Debug.Assert(a.Length == b.Length);
			float sum = 0;
			for (int i = 0; i < a.Length; i++) {
				sum += a[i] * b[i];
			}
			return sum;
		}

		public static int Dot(int[] a, int[] b, int prec) {
			Debug.Assert(a.Length == b.Length);
			long sum = 0;
			for (int i = 0; i < a.Length; i++) {
				sum += (long)a[i] * (long)b[i];
			}
			return (int)(sum >> prec);
		}

		private static float Dot(float[,] a, float[,] b, int arow, int bcol) {
			float sum = 0;
			int acol = a.GetLength(1);
			int brow = b.GetLength(0);
			Debug.Assert(acol == brow);
			for (int i = 0; i < acol; i++) {
				sum += a[arow, i] * b[i, bcol];
			}
			return sum;
		}

		public static void Mult(float[,] a, float[,] b, float[,] c) {
			int arow = a.GetLength(0);
			int acol = a.GetLength(1);
			int brow = b.GetLength(0);
			int bcol = b.GetLength(1);
			int crow = arow;
			int ccol = bcol;
			for (int j = 0; j < crow; j++) {
				for (int k = 0; k < ccol; k++) {
					c[j, k] = Dot(a, b, crow, ccol);
				}
			}
		}
	}

	public class SampleM {
		public float[] X;
		public float Y;

		public SampleM() {
		}

		public SampleM(float[] x, float y) {
			X = x;
			Y = y;
		}
		public void Clone(SampleM source) {
			X = new float[source.X.Length];
			for (int i = 0; i < X.Length; i++) {
				X[i] = source.X[i];
			}
			Y = source.Y;
		}

		public SampleM Clone() {
			SampleM copy = new SampleM();
			copy.X = new float[X.Length];
			for (int i = 0; i < X.Length; i++) {
				copy.X[i] = X[i];
			}
			copy.Y = Y;
			return copy;
		}
	}

	public class SampleS {
		public float X;
		public float Y;

		public SampleS() {
		}

		public SampleS(float x, float y) {
			X = x;
			Y = y;
		}
		public void Clone(SampleS source) {
			X = source.X;
			Y = source.Y;
		}

		public SampleS Clone() {
			SampleS copy = new SampleS();
			copy.X = X;
			copy.Y = Y;
			return copy;
		}
	}

	public class Model {
		List<BaseLayer> Layers;
		Random randomizer;

		public Model() {
			Layers = new List<BaseLayer>();
			randomizer = new Random();
		}

		public static void CloneSet(SampleS[] source, SampleS[] target) {
			for (int i = 0; i < source.Length; i++) {
				if (target[i] == null) target[i] = new SampleS();
				target[i].Clone(source[i]);
			}
		}

		public static void CloneSet(SampleM[] source, SampleM[] target) {
			for (int i = 0; i < source.Length; i++) {
				if (target[i] == null) target[i] = new SampleM();
				target[i].Clone(source[i]);
			}
		}

		public void AddLayer(BaseLayer layer) {
			// first layer in chain is automatically the input
			// last layer in chain is automatically the output
			if (Layers.Count > 0) {
				layer.Link(Layers[Layers.Count - 1]);
			} else {
				layer.Link(null);
			}
			Layers.Add(layer);
		}

		// performa a forward pass through the layers
		public float Forward(float x) {
			int numLayers = Layers.Count;
			for (int i = 0; i < numLayers; i++) {
				if (i == 0) {   // input layer
					((InputLayer)Layers[i]).Forward(x);
				} else {
					Layers[i].Forward();
				}
			}
			float y = Layers[numLayers - 1].ActivationArray[0];
			return y;
		}

		public float Forward(float[] x) {
			int numLayers = Layers.Count;
			for (int i = 0; i < numLayers; i++) {
				if (i == 0) {   // input layer
					((InputLayer)Layers[i]).Forward(x);
				} else {
					Layers[i].Forward();
				}
            }
			float y = Layers[numLayers-1].ActivationArray[0];
			return y;
		}

		public void Backward(float truth) {
			int numLayers = Layers.Count;
			for (int i = numLayers - 1; i >= 0; i--) {
				if (i == numLayers - 1) {   // output layer
					((OutputLayer)Layers[i]).Backward(truth);
				} else {
					Layers[i].Backward();
				}
			}
		}

		// multi-dim input
		public float TrainSet(SampleM[] points) {
			int setSize = points.Length;
			SampleM[] shuffle = new SampleM[setSize];
			CloneSet(points, shuffle);
			randomizer.Shuffle(shuffle);
			float err = 0;
			for (int i = 0; i < setSize; i++) {
				err += Train(shuffle[i].X, shuffle[i].Y);
			}
			return err / setSize;
		}
		public float Train(float[] x, float y) {
			Forward(x);
			float err = ((OutputLayer)Layers[Layers.Count - 1]).LossFunction(Layers[Layers.Count - 1].ActivationArray[0], y);
			Backward(y);
			return err;
		}

		// single-dim input
		public float TrainSet(SampleS[] points) {
			int setSize = points.Length;
			SampleS[] shuffle = new SampleS[setSize];
			CloneSet(points, shuffle);
			randomizer.Shuffle(shuffle);
			float err = 0;
			for (int i = 0; i < setSize; i++) {
				err += Train(points[i].X, points[i].Y);
			}
			return err / setSize;
		}
		public float Train(float x, float y) {
			Forward(x);
			int lastLayer = Layers.Count - 1;
			float err = ((OutputLayer)Layers[lastLayer]).LossFunction(Layers[lastLayer].ActivationArray[0], y);
			Backward(y);
			return err;
		}

		public void PredictSet(SampleS[] predict) {
			for (int i = 0; i < predict.Length; i++) {
				predict[i].Y = Forward(predict[i].X);
			}
		}

		public void PredictSet(SampleM[] predict) {
			for (int i = 0; i < predict.Length; i++) {
				predict[i].Y = Forward(predict[i].X);
			}
		}

		public string GetLayerWeights(int layer = -1) {
			StringBuilder sb = new StringBuilder();
			sb.AppendLine();
			if (layer >= 0) {
				sb.AppendLine(String.Format("Layer {0} Weights", layer));
				sb.AppendLine(Layers[0].GetWeights());
			} else {
				for (int n = 0; n < Layers.Count; n++) {
					sb.AppendLine(String.Format("Layer {0} Weights", n));
					sb.AppendLine(Layers[n].GetWeights());
				}
			}
			return sb.ToString();
		}

	}

	public class BaseLayer {
		public int InDims;   // input dims
		public int OutDims;   // output dims = # of neurons
		public float[] InputArray; // Input layer array
		public float[] OutputArray; // output layer array
		public float[] ActivationArray; // output activation array
		public float[][] WeightArray;
		public float[][] WeightDeltas;
		public float[] ActivationDeltas;
		public float[] Bias;
		public static bool QuantAwareTrain;
		public bool Quantized;
		public int[] InputArrayInt; // input layer array
		public int[] OutputArrayInt; // output layer array
		public int[] ActivationArrayInt; // output activation array
		public int[][] WeightArrayInt;
		public int[] BiasInt;
		public LayerType LayerID;
		public BiasType UseBias;
		public static Random Randomizer;
		public static int Prec = 24;    // fixed-point decimal precision
		public int QuantFactor = 1 << Prec; // float 1.0 = int 2^16, 16.16 fixed-point

		public ActivationType OutputActivation;
		public static float LearningRate;
		public BaseLayer UpLayer, DownLayer;
		public float[] Delta;

		public static bool QAT {        // Turn QAT on/off (affects all layers)
			set {
				QuantAwareTrain = value;
				if (QuantAwareTrain) {
					if (Activation.TanhLut == null) Activation.CreateTanhLut();
					if (Activation.SigmoidLut == null) Activation.CreateSigmoidLut();
				}
			}
			get {
				return QuantAwareTrain;
			}
		}

		public BaseLayer(int outDims, BiasType useBias) {
			LayerID = LayerType.Base;
			OutDims = outDims;
			OutputArray = new float[OutDims];
			ActivationArray = new float[OutDims];
			WeightArray = new float[OutDims][];
			Bias = new float[OutDims];
			Delta = new float[OutDims];
			ActivationDeltas = new float[OutDims];
			Quantized = false;
			OutputActivation = ActivationType.Linear;
			UseBias = useBias;
			OutputArrayInt = new int[OutDims];
			ActivationArrayInt = new int[OutDims];
			WeightArrayInt = new int[OutDims][];
			BiasInt = new int[OutDims];
		}

		public void Link(BaseLayer upLayer) {
			UpLayer = upLayer;
			if (UpLayer != null) {
				InDims = upLayer.OutDims;
				InputArray = UpLayer.ActivationArray;
				InputArrayInt = UpLayer.ActivationArrayInt;
			} else {	// for InputLayer only
				InputArray = new float[InDims];
				InputArrayInt = new int[InDims];
			}
			for (int c = 0; c < OutDims; c++) {
				WeightArray[c] = new float[InDims];
				WeightArrayInt[c] = new int[InDims];
			}
			float invsqrt = (float)(1f / Math.Sqrt(InDims));
			invsqrt = (float)Math.Min(0.5, invsqrt);
			if (Randomizer == null) Randomizer = new Random();
			for (int c = 0; c < OutDims; c++) {
				for (int i = 0; i < InDims; i++) {
					WeightArray[c][i] = (float)(Randomizer.NextDouble() * 2 * invsqrt) - invsqrt;
				}
			}
			switch (UseBias) {
				case BiasType.None:
					for (int c = 0; c < OutDims; c++) Bias[c] = 0; break;
				case BiasType.Shared:
					Bias[0] = (float)(Randomizer.NextDouble() * 2 * invsqrt) - invsqrt;
					for (int c = 1; c < OutDims; c++) Bias[c] = 0; break;
				case BiasType.Unique:
					for (int c = 0; c < OutDims; c++) {
						Bias[c] = (float)(Randomizer.NextDouble() * 2 * invsqrt) - invsqrt;
					}
					break;
			}
			WeightDeltas = new float[OutDims][];
			for (int c = 0; c < OutDims; c++) {
				WeightDeltas[c] = new float[InDims];
			}
		}

		public void Forward() {
			if (QuantAwareTrain) {
				if (!Quantized) Quantize();
				for (int c = 0; c < OutDims; c++) {
					OutputArrayInt[c] = Matrix.Dot(InputArrayInt, WeightArrayInt[c], Prec);
					switch (UseBias) {
						case BiasType.None: break;
						case BiasType.Shared: OutputArrayInt[c] += BiasInt[0]; break;
						case BiasType.Unique: OutputArrayInt[c] += BiasInt[c]; break;
					}
					ActivationArrayInt[c] = Activation.Activate(OutputActivation, OutputArrayInt[c]);
				}
				// update the FP arrays for use in backprop
				for (int c = 0; c < OutDims; c++) {
					// activation output is int: -127 ~ +127 for float -1 ~ +1
					OutputArray[c] = (float)OutputArrayInt[c] / (float)QuantFactor;
					ActivationArray[c] = (float)ActivationArrayInt[c] / (float)QuantFactor;
				}
			} else {
				for (int c = 0; c < OutDims; c++) {
					OutputArray[c] = Matrix.Dot(InputArray, WeightArray[c]);
					switch (UseBias) {
						case BiasType.None: break;
						case BiasType.Shared: OutputArray[c] += Bias[0]; break;
						case BiasType.Unique: OutputArray[c] += Bias[c]; break;
					}
					ActivationArray[c] = Activation.Activate(OutputActivation, OutputArray[c]);
				}
			}
		}

		public void Backward() {
			float grad;
			float sumErr;
			if (UpLayer != null) {
				for (int c = 0; c < OutDims; c++) {
					grad = Activation.Gradient(OutputActivation, ActivationArray[c]);
					ActivationDeltas[c] = grad * Delta[c];
				}
				for (int i = 0; i < InDims; i++) {
					sumErr = 0;
					for (int c = 0; c < OutDims; c++) {
						sumErr += ActivationDeltas[c] * WeightArray[c][i];
					}
					UpLayer.Delta[i] = sumErr;
				}
				AdjustWeights();
			}
			Quantized = false;
		}


		public void AdjustWeights() {
			for (int c = 0; c < OutDims; c++) {
				for (int i = 0; i < InDims; i++) {
					WeightArray[c][i] -= ActivationDeltas[c] * InputArray[i] * LearningRate;
				}
				switch (UseBias) {
					case BiasType.None: break;
					case BiasType.Shared: Bias[0] -= ActivationDeltas[c] * LearningRate; break;
					case BiasType.Unique: Bias[c] -= ActivationDeltas[c] * LearningRate; break;
				}
			}
		}

		public string GetWeights() {
			StringBuilder sb = new StringBuilder();
			float min = float.MaxValue;
			float max = float.MinValue;
			for (int c = 0; c < OutDims; c++) {
				sb.Append(String.Format("Neuron {0} Weights:", c));
				for (int i = 0; i < InDims; i++) {
					sb.Append(WeightArray[c][i].ToString());
					if (min > WeightArray[c][i]) min = WeightArray[c][i];
					if (max < WeightArray[c][i]) max = WeightArray[c][i];
				}
				sb.AppendLine();
				if (UseBias != BiasType.None) {
					sb.AppendLine(String.Format("Neuron {0} Bias: {1}", c, Bias[c].ToString()));
					if (min > Bias[c]) min = Bias[c];
					if (max < Bias[c]) max = Bias[c];
				}
			}
			sb.AppendLine(String.Format("Layer min value: {0}", min.ToString()));
			sb.AppendLine(String.Format("Layer max value: {0}", max.ToString()));
			sb.AppendLine(String.Format("Bias type: {0}", UseBias.ToString()));
			sb.AppendLine(String.Format("Activation type: {0}", OutputActivation.ToString()));
			return sb.ToString();
		}

		public void Quantize() {
			for (int c = 0; c < OutDims; c++) {
				for (int i = 0; i < InDims; i++) {
					WeightArrayInt[c][i] = (int)Math.Round(WeightArray[c][i] * QuantFactor);
				}
				BiasInt[c] = (int)Math.Round(Bias[c] * QuantFactor);
			}
			Quantized = true;
		}
	}

	public class InputLayer : BaseLayer {
		public InputLayer(int inDims) : base(inDims, BiasType.None) {
			InDims = inDims;
			LayerID = LayerType.Input;
			OutputActivation = ActivationType.Linear;
		}

		public void Forward(float[] x) {
			Array.Copy(x, InputArray, x.Length);
			Array.Copy(x, OutputArray, x.Length);
			Array.Copy(x, ActivationArray, x.Length);
			if (QuantAwareTrain) {
				int quantizedInput;
				for (int i = 0; i < x.Length; i++) {
					quantizedInput = (int)(x[i] * QuantFactor);
					InputArrayInt[i] = quantizedInput;
					OutputArrayInt[i] = quantizedInput;
					ActivationArrayInt[i] = quantizedInput;
				}
			}
		}

		public void Forward(float x) {
			InputArray[0] = x;
			OutputArray[0] = x;
			ActivationArray[0] = x;
			if (QuantAwareTrain) {
				int quantizedInput = (int)(x * QuantFactor);
				InputArrayInt[0] = quantizedInput;
				OutputArrayInt[0] = quantizedInput;
				ActivationArrayInt[0] = quantizedInput;
			}
		}

		public void Backward() {
			base.Backward();
		}
	}

	public class DenseLayer : BaseLayer {

		public DenseLayer(int outDims, ActivationType activation, BiasType useBias = BiasType.None) : base(outDims, useBias) {
			LayerID = LayerType.Dense;
			OutputActivation = activation;
		}

		public void Forward() {
			base.Forward();
		}

		public void Backward() {
			base.Backward();
		}
	}

	public class OutputLayer : BaseLayer {
		public OutputLayer(ActivationType activation) : base(1, BiasType.None) {
			LayerID = LayerType.Output;
			OutputActivation = activation;
		}

		public void Forward() {
			base.Forward();
		}

		public void Backward(float y) {
			Delta[0] = LossFunctionDx(ActivationArray[0], y);
			base.Backward();
		}

		// x = nn output
		// y = truth
		public virtual float LossFunction(float x, float y) {
			//float sign = (x - y) > 0 ? 1 : -1;
			//return (float)((x - y) * (x - y)) * sign;
			return (float)((x - y) * (x - y) * 0.5);
		}

		// x = nn output
		// y = truth
		public virtual float LossFunctionDx(float x, float y) {
			return (float)(x - y);
		}
	}
}
