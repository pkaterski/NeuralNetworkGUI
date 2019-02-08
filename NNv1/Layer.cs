using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNv1
{
    public class Layer
    {
        private int prevLayerSize;

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
        public int PrevLayerSize
        {
            get { return prevLayerSize; }
            set { prevLayerSize = value >= 0 ? value : 0; }
        }

        public Layer(int size, int prevLayerSize, Random r, double rScale, IActivation activation)
        {
            PrevLayerSize = prevLayerSize;
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
            PrevLayerSize = prevLayerSize;
            Activation = activation;
        }

        public double[] GetValues()
        {
            double[] result = new double[Size];
            for (int i = 0; i < Size; i++)
            {
                result[i] = Neurons[i].Value;
            }
            return result;
        }
    }
}
