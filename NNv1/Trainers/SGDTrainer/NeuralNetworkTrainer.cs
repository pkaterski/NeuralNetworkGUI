using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNv1
{
    /// <summary>
    /// A class wich trains a Neural Network based on Stochastic Gradient Descent
    /// The equations of the algorithm are from the resource:
    /// http://neuralnetworksanddeeplearning.com/chap1.html
    /// </summary>
    public class NeuralNetworkTrainer
    {
        // the layers containing the gradient of the previous update to the neural network
        // stored for applying momentum to the learning
        private LayerHelper[] prevLayers;

        public NeuralNetworkTrainer(NeuralNetwork nn)
        {
            prevLayers = InitLayers(nn);
        }

        public double UpdateMiniBatch(NeuralNetwork nn, (double[], double[])[] miniBatch, double eta, double momentum)
        {
            // create sum all all nablas in mini batch for Stochastic gradient descent
            LayerHelper[] nablaLayers = InitLayers(nn);
            foreach (var data in miniBatch)
            {
                double[] x = data.Item1;
                double[] y = data.Item2;

                //LayerHelper[] deltaNablaLayers = Backprop(nn, x, y);
                nablaLayers = Backprop(nn, x, y);

                //for (int i = 0; i < nablaLayers.Length; i++)
                //{
                //    for (int j = 0; j < nablaLayers[i].Neurons.Length; j++)
                //    {
                //        nablaLayers[i].Neurons[j].DB += deltaNablaLayers[i].Neurons[j].DB;
                //        for (int k = 0; k < nablaLayers[i].Neurons[j].DW.Length; k++)
                //        {
                //            nablaLayers[i].Neurons[j].DW[k] += deltaNablaLayers[i].Neurons[j].DW[k];
                //        }
                //    }
                //}

                // update weights and biases
                for (int i = 0; i < nn.Layers.Length; i++)
                {
                    for (int j = 0; j < nn.Layers[i].Neurons.Length; j++)
                    {
                        nn.Layers[i].Neurons[j].Bias -= (1 - momentum) * nablaLayers[i].Neurons[j].DB * eta
                            / 1 + momentum * prevLayers[i].Neurons[j].DB;

                        // update the prevlayer
                        prevLayers[i].Neurons[j].DB = nablaLayers[i].Neurons[j].DB * eta / 1;

                        for (int k = 0; k < nn.Layers[i].Neurons[j].Weights.Length; k++)
                        {
                            nn.Layers[i].Neurons[j].Weights[k] -= (1 - momentum) * nablaLayers[i].Neurons[j].DW[k] * eta
                                / 1 + momentum * prevLayers[i].Neurons[j].DW[k];

                            // update the prev layer
                            prevLayers[i].Neurons[j].DW[k] = nablaLayers[i].Neurons[j].DW[k] * eta / 1;
                        }
                    }
                }
            }

            // return the error
            double errorValue = 0;
            for (int i = 0; i < miniBatch.Length; i++)
            {
                errorValue += Cost(nn.Forward(miniBatch[i].Item1), miniBatch[i].Item2);
            }

            return errorValue / miniBatch.Length;
        }

        public static LayerHelper[] Backprop(NeuralNetwork nn, double[] input, double[] targetOutput)
        {
            // handle errors
            if (nn.Layers == null || nn.Layers.Length < 2)
                throw new ArgumentException("invalid neural network", "original");

            if (nn.Layers[0].Neurons.Length != input.Length)
                throw new ArgumentException("nerual network input and x mismatch", "original");

            if (nn.Layers[nn.Layers.Length - 1].Neurons.Length != targetOutput.Length)
                throw new ArgumentException("nerual network output and y mismatch", "original");


            // set up the helper layers
            LayerHelper[] layerHelpers = InitLayers(nn);

            // Input x: Set the corresponding activation a1 for the input layer.
            for (int i = 0; i < nn.Layers[0].Neurons.Length; i++)
            {
                nn.Layers[0].Neurons[i].Output = input[i];
                layerHelpers[0].Neurons[i].A = input[i];
            }

            // Feedforward: For each l=2,3,…,L compute zl=wla(l−1)+bl and al=σ(zl).
            for (int i = 1; i < nn.Layers.Length; i++)
            {
                for (int j = 0; j < nn.Layers[i].Neurons.Length; j++)
                {
                    double total = 0;
                    for (int k = 0; k < nn.Layers[i].Neurons[j].Weights.Length; k++)
                    {
                        total += nn.Layers[i - 1].Neurons[k].Output * nn.Layers[i].Neurons[j].Weights[k];
                    }
                    total += nn.Layers[i].Neurons[j].Bias;
                    layerHelpers[i].Neurons[j].Z = total; // Z of helper
                    total = nn.Layers[i].Activation.Activate(total);
                    layerHelpers[i].Neurons[j].A = total; // A = σ(z)
                    nn.Layers[i].Neurons[j].Output = total;
                }
            }

            // Output error δL: Compute the vector δL=∇aC⊙σ′(zL).
            int L = layerHelpers.Length - 1;

            for (int i = 0; i < layerHelpers[L].Neurons.Length; i++)
            {
                layerHelpers[L].Neurons[i].Delta =
                    CostPrime(layerHelpers[L].Neurons[i].A, targetOutput[i])
                    * nn.Layers[L].Activation.ActivatePrime(layerHelpers[L].Neurons[i].Z);

                // Output: The gradient of the cost function is given by ∂C∂wljk=al−1kδlj and ∂C∂blj=δlj.
                layerHelpers[L].Neurons[i].DB = layerHelpers[L].Neurons[i].Delta;
                for (int j = 0; j < layerHelpers[L - 1].Neurons.Length; j++)
                {
                    layerHelpers[L].Neurons[i].DW[j] =
                        layerHelpers[L - 1].Neurons[j].A * layerHelpers[L].Neurons[i].Delta;
                }
            }

            // Backpropagate the error: For each l=L−1,L−2,…,2 compute δl=((wl+1)Tδl+1)⊙σ′(zl).
            // δlj=∑k w(l+1)kj δ(l+1)kσ′(zlj).
            for (int i = layerHelpers.Length - 2; i > 0; i--)
            {
                for (int j = 0; j < layerHelpers[i].Neurons.Length; j++)
                {
                    double total = 0;
                    for (int k = 0; k < layerHelpers[i + 1].Neurons.Length; k++)
                    {
                        total += layerHelpers[i + 1].Neurons[k].Delta
                            * nn.Layers[i + 1].Neurons[k].Weights[j]
                            * nn.Layers[i]
                              .Activation.ActivatePrime(layerHelpers[i].Neurons[j].Z);
                    }

                    layerHelpers[i].Neurons[j].Delta = total;

                    // Output: The gradient of the cost function is given by ∂C∂wljk=al−1kδlj and ∂C∂blj=δlj.
                    layerHelpers[i].Neurons[j].DB = layerHelpers[i].Neurons[j].Delta;
                    for (int k = 0; k < layerHelpers[i - 1].Neurons.Length; k++)
                    {
                        layerHelpers[i].Neurons[j].DW[k] =
                            layerHelpers[i - 1].Neurons[k].A * layerHelpers[i].Neurons[j].Delta;
                    }
                }
            }


            return layerHelpers;
        }

        private static double CostPrime(double output, double y)
        {
            return output - y;
        }

        private static double Cost(double[] output, double[] targetOutput)
        {
            if (output.Length != targetOutput.Length)
                throw new ArgumentException("Cannot compute error. NN output and y dims mismatch", "original");

            double total = 0;
            for (int i = 0; i < output.Length; i++)
            {
                total += (output[i] - targetOutput[i]) * (output[i] - targetOutput[i]);
            }

            return total / 2;
        }

        private static LayerHelper[] InitLayers(NeuralNetwork nn)
        {
            LayerHelper[] layerHelpers = new LayerHelper[nn.Layers.Length];

            for (int i = 0; i < nn.Layers.Length; i++)
            {
                NeuronHelper[] neurons = new NeuronHelper[nn.Layers[i].Neurons.Length];
                for (int j = 0; j < nn.Layers[i].Neurons.Length; j++)
                {
                    neurons[j] = new NeuronHelper(nn.Layers[i].Neurons[j].InputCount);
                }
                layerHelpers[i] = new LayerHelper(neurons);
            }

            return layerHelpers;
        }
    }
}
