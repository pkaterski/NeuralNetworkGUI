using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNv1
{
     public class Sigmoid : IActivation
     {
            public double Y { get; set; }

            public double Activate(double x)
            {
                return 1d / (1d + Math.Exp(-Y * x));
            }

            public double ActivatePrime(double x)
            {
                double s = Activate(x);
                return Y * s * (1 - s);
            }

            public Sigmoid(double y = 1)
            {
                Y = y;
            }
    }
}
