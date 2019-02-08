using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkPrev.Controller
{
    interface IManager
    {
        double Train((double[], double[])[] miniBatch);
    }
}
