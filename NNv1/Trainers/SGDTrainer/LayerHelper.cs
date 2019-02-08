using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNv1
{
    public class LayerHelper
    {

        public NeuronHelper[] Neurons { get; set; }

        // the number of neurons in the layer
        public int Size
        {
            get
            {
                if (Neurons != null) return Neurons.Length;
                else return 0;
            }
        }

        public LayerHelper(NeuronHelper[] neurons)
        {
            Neurons = new NeuronHelper[neurons.Length];
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new NeuronHelper(neurons[i]);
            }
        }

        public LayerHelper() { }
    }
}
