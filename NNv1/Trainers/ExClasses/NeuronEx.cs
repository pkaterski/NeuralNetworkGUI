using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNv1
{
    public class NeuronEx
    {
        public double[] Weights { get; set; }
        public double Bias { get; set; }

        private int inputCount;
        public int InputCount
        {
            get
            {
                return inputCount;
            }
        }

        private double output;
        public double Output { get { return output; } }


        //*** training helpers
        public double Error { get; internal set; }
        public double BiasDelta { get; internal set; }
        public double[] WeightsDelta { get; set; }

        public NeuronEx(Neuron neuron)
        {
            inputCount = neuron.InputCount;
            Weights = new double[inputCount];
            WeightsDelta = new double[inputCount];
            for (int i = 0; i < inputCount; i++)
            {
                Weights[i] = neuron.Weights[i];
                WeightsDelta[i] = 0.0;
            }

            Bias = neuron.Bias;
            this.output = neuron.Output;

        }

    }
}
