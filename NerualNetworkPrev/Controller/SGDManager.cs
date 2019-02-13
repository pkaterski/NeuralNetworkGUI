using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNv1;

namespace NeuralNetworkPrev.Controller
{
    // the manager is responsible for training a NN
    class SGDManager : IManager
    {
        private NeuralNetworkTrainer trainer;
        private NeuralNetwork nn;
        public double eta;
        public double momentum;

        public SGDManager(NeuralNetwork nn)
        {
            trainer = new NeuralNetworkTrainer(nn);
            this.nn = nn;
        }

        public double Train((double[], double[])[] miniBatch)
        {
            return trainer.UpdateMiniBatch(nn, miniBatch, eta, momentum);
        }

        public double UpdateMiniBatch(NeuralNetwork nn, (double[], double[])[] miniBatch, double eta, double momentum)
        {
            return trainer.UpdateMiniBatch(nn, miniBatch, eta, momentum);
        }
    }
}
