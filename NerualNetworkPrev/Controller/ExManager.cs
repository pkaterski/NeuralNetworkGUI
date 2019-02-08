using NNv1;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkPrev.Controller
{
    class ExManager : IManager
    {
        private NeuralNetworkTrainerEx trainer;
        public double eta = 0.5;
        public double momentum = 0.1;

        public ExManager(NeuralNetwork nn)
        {
            trainer = new NeuralNetworkTrainerEx(nn);
        }

        public double UpdateMiniBatch((double[], double[])[] miniBatch, double eta, double momentum)
        {
            return trainer.UpdateMiniBatch(miniBatch, eta, momentum);
        }

        public double Train((double[], double[])[] miniBatch)
        {
            return trainer.UpdateMiniBatch(miniBatch, eta, momentum);
        }

    }
}
