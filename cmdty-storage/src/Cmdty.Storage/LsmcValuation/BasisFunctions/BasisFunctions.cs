#region License
// Copyright (c) 2020 Jake Fowler
//
// Permission is hereby granted, free of charge, to any person 
// obtaining a copy of this software and associated documentation 
// files (the "Software"), to deal in the Software without 
// restriction, including without limitation the rights to use, 
// copy, modify, merge, publish, distribute, sublicense, and/or sell 
// copies of the Software, and to permit persons to whom the 
// Software is furnished to do so, subject to the following 
// conditions:
//
// The above copyright notice and this permission notice shall be 
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
// OTHER DEALINGS IN THE SOFTWARE.
#endregion

using System;
using System.Collections.Generic;
using System.Linq;

namespace Cmdty.Storage
{
    public static class BasisFunctions
    {
        public static BasisFunction Ones
        {
            get
            {
                static void Ones(ReadOnlyMemory<double>[] markovFactors, ReadOnlySpan<double> spotPriceBySim,
                    Span<double> designMatrixCol)
                {
                    for (int i = 0; i < designMatrixCol.Length; i++)
                        designMatrixCol[i] = 1.0;
                }
                return Ones;
            }
        }

        public static BasisFunction SpotPricePower(int power)
        {
            void BasisFunc(ReadOnlyMemory<double>[] markovFactors, ReadOnlySpan<double> spotPriceBySim,
                Span<double> designMatrixCol)
            {
                for (int i = 0; i < designMatrixCol.Length; i++)
                    designMatrixCol[i] = Math.Pow(spotPriceBySim[i], power);
            }
            return BasisFunc;
        }

        public static BasisFunction MarkovFactorPower(int markovFactor, int power)
        {
            void BasisFunc(ReadOnlyMemory<double>[] markovFactors, ReadOnlySpan<double> spotPriceBySim,
                Span<double> designMatrixCol)
            {
                ReadOnlySpan<double> markovFactorBySim = markovFactors[markovFactor].Span;
                for (int i = 0; i < designMatrixCol.Length; i++)
                    designMatrixCol[i] = Math.Pow(markovFactorBySim[i], power);
            }
            return BasisFunc;
        }

        public static IEnumerable<BasisFunction> MarkovFactorAllPositiveIntegerPowersUpTo(int markovFactor, int maxPower)
        {
            MaxPowerPrecondition(maxPower);
            for (int i = 0; i < maxPower; i++)
                yield return MarkovFactorPower(markovFactor, i + 1);
        }

        private static void MaxPowerPrecondition(int maxPower)
        {
            if (maxPower < 1)
                throw new ArgumentException("Maximum power must be greater than zero.");
        }

        public static IEnumerable<BasisFunction> AllMarkovFactorAllPositiveIntegerPowersUpTo(int maxPower, int numMarkovFactors)
        {
            MaxPowerPrecondition(maxPower);
            for (int i = 0; i < numMarkovFactors; i++)
                foreach (BasisFunction basisFunction in MarkovFactorAllPositiveIntegerPowersUpTo(i, maxPower))
                    yield return basisFunction;
        }

        public static BasisFunction Generic(int spotPower, Dictionary<int, int> markovFactorPowers)
        {
            if (spotPower < 0)
                throw new ArgumentException("Spot power must be non-negative.");
            if (markovFactorPowers.Any(pair => pair.Value < 0))
                throw new ArgumentException("Markov factor powers must be non-negative.");

            BasisFunction basisFunction;

            if (markovFactorPowers.Count == 0) // Just a spot power
            {
                basisFunction = SpotPricePower(spotPower);
            }
            else
            {
                basisFunction = (markovFactors, spotPriceBySim, designMatrixCol) =>
                {
                    if (spotPower == 0)
                    {
                        for (int simIndex = 0; simIndex < designMatrixCol.Length; simIndex++)
                            designMatrixCol[simIndex] = 1.0;
                    }
                    else
                    {
                        for (int simIndex = 0; simIndex < designMatrixCol.Length; simIndex++)
                            designMatrixCol[simIndex] = Math.Pow(spotPriceBySim[simIndex], spotPower); // TODO replace with power function optimised for integer?
                    }

                    foreach (int markovIndex in markovFactorPowers.Keys)
                    {
                        int markovPower = markovFactorPowers[markovIndex];
                        if (markovPower > 0)
                        {
                            ReadOnlySpan<double> markovFactorSims = markovFactors[markovIndex].Span;
                            for (int simIndex = 0; simIndex < designMatrixCol.Length; simIndex++)
                            {
                                designMatrixCol[simIndex] *= Math.Pow(markovFactorSims[simIndex], markovPower); // TODO replace with power function optimised for integer?
                            }
                        }
                    }
                };
            }
            return basisFunction;
        }
        
    }
}