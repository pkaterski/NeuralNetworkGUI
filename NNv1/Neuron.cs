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
        public double Output { get; set; }

        public int InputCount
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
        /// Constructor which generates random values of the weights and biases of the neuron
        /// scaled at some value
        /// </summary>
        /// <param name="inputCount">The number of inputs in the neuron</param>
        /// <param name="r">A random object for generating random numbers</param>
        /// <param name="scale">The scale at which the weights will be initiallized</param>
        public Neuron(int inputCount, Random r, double scale = 1)
        {
            Weights = new double[inputCount];
            for (int i = 0; i < inputCount; i++)
            {
                Weights[i] = r.NextDouble() * scale;
            }

            Bias = r.NextDouble();
            Output = 0;
        }

        /// <summary>
        /// Constructor which manually sets the weights and biases
        /// </summary>
        /// <param name="weights"></param>
        /// <param name="bias"></param>
        public Neuron(double[] weights, double bias)
        {
            Weights = new double[weights.Length];
            for (int i = 0; i < weights.Length; i++)
            {
                Weights[i] = weights[i];
            }
            Bias = bias;
        }

        /// <summary>
        /// Copy constructor
        /// </summary>
        /// <param name="neuron"></param>
        public Neuron(Neuron neuron) : this(neuron.Weights, neuron.Bias)
        { }
    }
}
