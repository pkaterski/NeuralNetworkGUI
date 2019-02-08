using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNv1
{
    public class NeuralNetworkTrainerEx
    {
        private NeuralNetwork nn = null;

        private LayerEx[] layerHelpersEx = null;

        // learning rate
        private double learningRate = 0.25;
        // momentum
        private double momentum = 0.1;

        public double LearningRate
        {
            get { return learningRate; }
            set { learningRate = value; }
        }
        public double Momentum
        {
            get { return momentum; }
            set { momentum = value; }
        }


        public NeuralNetworkTrainerEx(NeuralNetwork nn)
        {
            this.nn = nn;
        }

        public double UpdateMiniBatch((double[], double[])[] miniBatch, double eta, double momentum)
        {
            learningRate = eta;
            Momentum = momentum;
            double error = 0.0;

            foreach (var data in miniBatch)
            {
                double[] inputs = data.Item1;
                double[] targets = data.Item2;

                error += Run(inputs, targets);

            }
            return error;
        }


        public double Run(double[] input, double[] output)
        {
            // compute the network's output
            nn.Forward(input);

            // set up the helper layers
            layerHelpersEx = InitLayersEx(nn);

            // calculate network error
            double error = CalculateErrorEx(output);

            // calculate weights updates
            CalculateUpdatesEx(input);

            // update the network
            UpdateNetworkEx();

            return error;
        }


        private LayerEx[] InitLayersEx(NeuralNetwork nn)
        {

            // Create layer helpers -> one layer less then in the network!
            // BTW = we do not need this first(inputs) layer in the network!? 
            LayerEx[] layerHelpers = new LayerEx[nn.Layers.Length - 1];

            for (int i = 1; i < nn.Layers.Length; i++)
            {
                NeuronEx[] neuronsEx = new NeuronEx[nn.Layers[i].Neurons.Length];
                
                //Make a copy of the network neurons to the neuronsEx
                //neuronEx has some additional properties needed for the training
                for (int j = 0; j < nn.Layers[i].Neurons.Length; j++)
                {
                    neuronsEx[j] = new NeuronEx(nn.Layers[i].Neurons[j]);
                }
                layerHelpers[i - 1] = new LayerEx();
                layerHelpers[i - 1].NeuronsEx = neuronsEx;
                //layerHelpers[i - 1].PrevLayerSize = nn.Layers[i].PrevLayerSize;
            }

            return layerHelpers;
        }

        private double CalculateErrorEx(double[] desiredOutput)
        {

            // current and the next layers
            LayerEx layer, layerNext;

            // error values
            double error = 0, e, sum;
            // neuron's output value
            double output;
            // layers count
            int layersCount = layerHelpersEx.Length; // nn.Layers.Length;


            // calculate error values for the last layer first
            layer = layerHelpersEx[layersCount - 1];

            double alfa = 1.0; //***AKG

            for (int i = 0; i < layer.NeuronsEx.Length; i++)
            {
                output = layer.NeuronsEx[i].Output;
                // error of the neuron
                e = desiredOutput[i] - output;
                // error multiplied with activation function's derivative
                //****errors[i] = e * function.Derivative2(output);
                layer.NeuronsEx[i].Error = e * (double)(alfa * output * (1 - output)); // Derivative2
                // squre the error and sum it
                error += (e * e);
            }

            // calculate error values for other layers
            for (int j = layersCount - 2; j >= 0; j--)
            {

                layer = layerHelpersEx[j];
                layerNext = layerHelpersEx[j + 1];

                // for all neurons of the layer
                for (int i = 0; i < layer.NeuronsEx.Length; i++)
                {
                    sum = 0.0;
                    // for all neurons of the next layer
                    for (int k = 0; k < layerNext.NeuronsEx.Length; k++)
                    {
                        sum += layerNext.NeuronsEx[k].Error * layerNext.NeuronsEx[k].Weights[i];
                    }
                    //***errors[i] = sum * function.Derivative2(layer.Neurons[i].Output);
                    layer.NeuronsEx[i].Error = sum * (double)(alfa * layer.NeuronsEx[i].Output * (1 - layer.NeuronsEx[i].Output)); //Derivative2
                }
            }



            // return squared error of the last layer divided by 2
            return error / 2.0;

        }


        private void UpdateNetworkEx()
        {

            // current neuron
            Neuron neuron;
            // current layer
            Layer layer;
            // current neuron
            NeuronEx neuronEx;
            // current layer
            LayerEx layerEx;


            // for each layer of the network
            for (int i = 1; i < nn.Layers.Length; i++)
            {
                layer = nn.Layers[i];
                layerEx = layerHelpersEx[i - 1];

                // for each neuron of the layer
                for (int j = 0; j < layer.Neurons.Length; j++)
                {
                    neuron = layer.Neurons[j] as Neuron;
                    neuronEx = layerEx.NeuronsEx[j];

                    // for each weight of the neuron
                    for (int k = 0; k < neuron.Weights.Length; k++)
                    {
                        // update weight
                        neuron.Weights[k] += neuronEx.WeightsDelta[k];
                    }
                    // update treshold
                    neuron.Bias += neuronEx.BiasDelta;
                }
            }

        }

        private void CalculateUpdatesEx(double[] input)
        {

            // current neuron
            NeuronEx neuron;
            // current and previous layers
            LayerEx layer, layerPrev;

            // 1 - calculate updates for the first layer
            layer = layerHelpersEx[0];

            // cache for frequently used values
            double varMomentum = learningRate * momentum;
            double var1Momentum = learningRate * (1 - momentum);
            double varError;

            // for each neuron of the layer
            for (int i = 0; i < layer.NeuronsEx.Length; i++)
            {
                neuron = layer.NeuronsEx[i];
                varError = neuron.Error * var1Momentum;

                // for each weight of the neuron
                for (int j = 0; j < neuron.WeightsDelta.Length; j++)
                {
                    // calculate weight delta
                    neuron.WeightsDelta[j] = varMomentum * neuron.WeightsDelta[j] + varError * input[j];
                }

                // calculate bias update
                neuron.BiasDelta = varMomentum * neuron.BiasDelta + varError;
            }

            // 2 - for all other layers
            for (int k = 1; k < layerHelpersEx.Length; k++)
            {
                layerPrev = layerHelpersEx[k - 1];
                layer = layerHelpersEx[k];

                // for each neuron of the layer
                for (int i = 0; i < layer.NeuronsEx.Length; i++)
                {
                    neuron = layer.NeuronsEx[i];
                    varError = neuron.Error * var1Momentum;

                    // for each synapse of the neuron
                    for (int j = 0; j < neuron.WeightsDelta.Length; j++)
                    {
                        // calculate weight delta
                        neuron.WeightsDelta[j] = varMomentum * neuron.WeightsDelta[j] + varError * layerPrev.NeuronsEx[j].Output;
                    }

                    // calculate bias delta
                    neuron.BiasDelta = varMomentum * neuron.BiasDelta + varError;
                }
            }


        }



    }
}
