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
using Xunit;

namespace Cmdty.Storage.Test
{
    public sealed class PowerMonomialBuilderTest
    {

        [Fact]
        [Trait("Category", "Lsmc.BasisFunctions")]
        public void CreateBasisFunctionFromSim_AsExpected()
        {
            BasisFunction basis = Sim.Spot * Sim.Spot * Sim.X1 * Sim.X0.Pow(3);

            const int numSims = 3;
            var spotPriceSims = new[] { 25.69, 21.88, 16.78 };
            var markovFactors = new ReadOnlyMemory<double>[]
            {
                new[]{ 0.56, 0.12, 1.55},
                new[]{ 1.08, 2.088, 0.988}
            };

            var result = new double[numSims];
            basis(markovFactors, spotPriceSims, result);

            for (int i = 0; i < numSims; i++)
            {
                double expectedBasis = spotPriceSims[i] * spotPriceSims[i] * markovFactors[1].Span[i] * Math.Pow(markovFactors[0].Span[i], 3);
                Assert.Equal(expectedBasis, result[i]);
            }
        }

    }
}
