using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNv1
{
    public class Neuron
    {
        public double[] Weights { get; set; }
        public double Bias { get; set; }
        public double Value { get; set; }
        public int PrevLayerSize
        {
            get
            {
                if (Weights != null) return Weights.Length;
                else return 0;
            }
        }

        public Neuron()
        {
        }

        /// <summary>
        /// Constructor which generates random values
        /// </summary>
        /// <param name="prevLayerSize">The Sizes of the previous layer</param>
        /// <param name="r">A random object for generating random numbers</param>
        /// <param name="rScale">The scale at which the weights will be initiallized</param>
        public Neuron(int prevLayerSize, Random r, double rScale)
        {
            Weights = new double[prevLayerSize];
            for (int i = 0; i < prevLayerSize; i++)
            {
                Weights[i] = r.NextDouble() * rScale;
            }

            Bias = r.NextDouble();
            Value = 0;
        }

        public Neuron(double[] weights, double bias)
        {
            Weights = new double[weights.Length];
            for (int i = 0; i < weights.Length; i++)
            {
                Weights[i] = weights[i];
            }
            Bias = bias;
        }

        public Neuron(Neuron neuron) : this(neuron.Weights, neuron.Bias)
        { }
    }
}
