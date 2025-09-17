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

using System.Collections.Generic;
using System.Linq;

namespace Cmdty.Storage
{
    public sealed class PowerMonomialBuilder
    {
        public int SpotPower { get; }
        public Dictionary<int, int> MarkovFactorPowers { get; }

        public PowerMonomialBuilder(int spotPower, Dictionary<int, int> markovFactorPowers)
        {
            SpotPower = spotPower;
            MarkovFactorPowers = markovFactorPowers;
        }

        public PowerMonomialBuilder(int spotPower) : this(spotPower, new Dictionary<int, int>()) { }

        public static PowerMonomialBuilder operator *(PowerMonomialBuilder builder1, PowerMonomialBuilder builder2)
        {
            int spotPower = builder1.SpotPower + builder2.SpotPower;
            Dictionary<int, int> markovPowers =
                builder1.MarkovFactorPowers.ToDictionary(pair => pair.Key, pair => pair.Value);
            foreach (int markovIndex in builder2.MarkovFactorPowers.Keys)
            {
                if (markovPowers.ContainsKey(markovIndex))
                    markovPowers[markovIndex] = markovPowers[markovIndex] + builder2.MarkovFactorPowers[markovIndex];
                else
                    markovPowers[markovIndex] = builder2.MarkovFactorPowers[markovIndex];
            }
            return new PowerMonomialBuilder(spotPower, markovPowers);
        }

        public PowerMonomialBuilder Pow(int power)
        {
            int spotPower = SpotPower * power;
            Dictionary<int, int> markovPowers =
                MarkovFactorPowers.ToDictionary(pair => pair.Key, pair => pair.Value * power);
            return new PowerMonomialBuilder(spotPower, markovPowers);
        }

        public static implicit operator BasisFunction(PowerMonomialBuilder builder) =>
            BasisFunctions.Generic(builder.SpotPower, builder.MarkovFactorPowers);

        public static implicit operator BasisFunctionsBuilder(PowerMonomialBuilder builder) => new BasisFunctionsBuilder(builder); // TODO could replace this with implicit case on BasisFunctionsBuilder

        public static BasisFunctionsBuilder operator +(PowerMonomialBuilder builder1, PowerMonomialBuilder builder2)
        {
            return new BasisFunctionsBuilder(new BasisFunction[] { builder1, builder2 });
        }
    }
}