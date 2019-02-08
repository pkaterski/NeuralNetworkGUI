using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNv1
{
    public class Layer
    {
        public IActivation Activation { get; set; }

        public Neuron[] Neurons { get; set; }
        public int Size
        {
            get
            {
                if (Neurons != null) return Neurons.Length;
                else return 0;
            }
        }

        public Layer(int size, int prevLayerSize, Random r, double rScale, IActivation activation)
        {
            Neurons = new Neuron[size];
            for (int i = 0; i < size; i++)
            {
                Neurons[i] = new Neuron(prevLayerSize, r, rScale);
            }
            Activation = activation;
        }

        public Layer(Neuron[] neurons, int prevLayerSize, IActivation activation)
        {
            Neurons = new Neuron[neurons.Length];
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new Neuron(neurons[i]);
            }
            Activation = activation;
        }

        public double[] GetValues()
        {
            double[] result = new double[Size];
            for (int i = 0; i < Size; i++)
            {
                result[i] = Neurons[i].Output;
            }
            return result;
        }
    }
}
