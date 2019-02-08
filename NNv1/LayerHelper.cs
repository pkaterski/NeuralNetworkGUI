using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNv1
{
    public class LayerHelper
    {

        private int prevLayerSize;

        public NeuronHelper[] Neurons { get; set; }
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

        public LayerHelper(NeuronHelper[] neurons, int prevLayerSize)
        {
            Neurons = new NeuronHelper[neurons.Length];
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new NeuronHelper(neurons[i]);
            }
            PrevLayerSize = prevLayerSize;
        }

        public LayerHelper() { }
    }
}
