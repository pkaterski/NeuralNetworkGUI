using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNv1
{
    public class NeuralNetwork
    {
        public Layer[] Layers;

        public NeuralNetwork(Layer[] layers)
        {
            Layers = layers;
        }

        public NeuralNetwork(int[] sizes, double rScale = 1)
        {
            Layers = new Layer[sizes.Length];

            Sigmoid sigmoid = new Sigmoid();
            Random r = new Random();

            Layers[0] = new Layer(sizes[0], 0, r, rScale, sigmoid);
            for (int i = 1; i < sizes.Length; i++)
            {
                Layers[i] = new Layer(sizes[i], sizes[i - 1], r, rScale, sigmoid);
            }
        }

        public NeuralNetwork(int[] sizes, Random r, double rScale, IActivation[] activations)
        {
            if (sizes.Length != activations.Length)
                throw new ArgumentException("length of layers and length of activations don't match", "original");

            Layers = new Layer[sizes.Length];

            Layers[0] = new Layer(sizes[0], 0, r, rScale, activations[0]);
            for (int i = 1; i < sizes.Length; i++)
            {
                Layers[i] = new Layer(sizes[i], sizes[i - 1], r, rScale, activations[i]);
            }
        }

        public NeuralNetwork(int[] sizes, Random r, double rScale)
        {
            Layers = new Layer[sizes.Length];

            IActivation sigmoid = new Sigmoid();

            Layers[0] = new Layer(sizes[0], 0, r, rScale, sigmoid);

            for (int i = 0; i < Layers[0].Neurons.Length; i++)
            {
                Layers[0].Neurons[i].Bias = 0;
            }

            for (int i = 1; i < sizes.Length; i++)
            {
                Layers[i] = new Layer(sizes[i], sizes[i - 1], r, rScale, sigmoid);
            }
        }

        public double[] Forward(double[] input)
        {
            if (Layers[0] == null)
                throw new ArgumentException("No Input Layer", "original");

            if (input.Length != Layers[0].Size)
                throw new ArgumentException("input and layer0 dim mismatch", "original");


            for (int i = 0; i < input.Length; i++)
            {
                Layers[0].Neurons[i].Value = input[i];
            }

            double[] result = new double[Layers.Length];

            // cycle through all the layers
            for (int i = 1; i < Layers.Length; i++)
            {
                // cycle through every neuron in the current layer
                for (int j = 0; j < Layers[i].Neurons.Length; j++)
                {
                    // get the wighted sum of the neurons in the previous layer
                    double total = 0;
                    for (int k = 0; k < Layers[i - 1].Neurons.Length; k++)
                    {
                        total += Layers[i - 1].Neurons[k].Value * Layers[i].Neurons[j].Weights[k];
                    }
                    total += Layers[i].Neurons[j].Bias;
                    total = Layers[i].Activation.Activate(total);
                    Layers[i].Neurons[j].Value = total;
                }
            }

            // return the values of the last layer
            return Layers[Layers.Length - 1].GetValues();
        }

        public override string ToString()
        {
            string result = "";

            for (int i = 0; i < Layers.Length; i++)
            {
                result += "Layer " + i + "\n";
                for (int j = 0; j < Layers[i].Neurons.Length; j++)
                {
                    result += "\tNeuron " + j + ": \n";
                    for (int k = 0; k < Layers[i].Neurons[j].Weights.Length; k++)
                    {
                        result += "\t\tWeight " + k + ": " + Layers[i].Neurons[j].Weights[k] + "\n";
                    }
                    result += "\t\tBias " + ": " + Layers[i].Neurons[j].Bias + "\n";
                }
            }

            return result;
        }
    }
}
