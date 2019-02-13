using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNv1
{
    // a helper class for storing the data necessary for learning
    public class NeuronHelper
    {
        public double Delta { get; set; }
        public double A { get; set; }
        public double Z { get; set; }
        public double DB { get; set; } // ∂C∂b
        public double[] DW { get; set; } // ∂C∂wk

        public int InputCount
        {
            get
            {
                if (DW != null) return DW.Length;
                else return 0;
            }
        }

        public NeuronHelper()
        {
        }

        public NeuronHelper(double delta, double a, double z, double dB, double[] dW)
        {
            Delta = delta;
            A = a;
            Z = z;
            DB = dB;
            DW = new double[dW.Length];
            for (int i = 0; i < dW.Length; i++)
            {
                DW[i] = dW[i];
            }
        }

        public NeuronHelper(NeuronHelper n) : this(n.Delta, n.A, n.Z, n.DB, n.DW) { }

        public NeuronHelper(int inputCount)
        {
            DW = new double[inputCount];
        }

    }
}
