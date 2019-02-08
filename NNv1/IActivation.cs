using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNv1
{
    public interface IActivation
    {
        double Activate(double x);
        double ActivatePrime(double x);
    }
}
