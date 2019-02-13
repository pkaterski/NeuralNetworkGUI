using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNv1
{
    public class Relu : IActivation
    {
        public double Activate(double x)
        {
            return Math.Max(0, x);
        }

        public double ActivatePrime(double x)
        {
            return x >= 0 ? 1 : 0;
        }
    }
}
