using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNv1;

namespace NeuralNetworkPrev.Controller
{
    // the manager is responsible for training a NN
    class Manager
    {
        private NeuralNetworkTrainer trainer;

        public Manager(NeuralNetwork nn)
        {
            trainer = new NeuralNetworkTrainer(nn);
        }

        public double UpdateMiniBatch(NeuralNetwork nn, (double[], double[])[] miniBatch, double eta, double momentum)
        {
            return trainer.UpdateMiniBatch(nn, miniBatch, eta, momentum);
        }
    }
}
