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

namespace Cmdty.Storage
{
    public static class Sim
    {
        public static PowerMonomialBuilder Spot => new PowerMonomialBuilder(1);
        public static PowerMonomialBuilder S => new PowerMonomialBuilder(1);
        public static PowerMonomialBuilder Factor(int factorIndex)
            => new PowerMonomialBuilder(0, new Dictionary<int, int>() { { factorIndex, 1 } });
        public static PowerMonomialBuilder X0 => Factor(0);
        public static PowerMonomialBuilder X1 => Factor(1);
        public static PowerMonomialBuilder X2 => Factor(2);
        public static PowerMonomialBuilder X3 => Factor(3);
        public static PowerMonomialBuilder X4 => Factor(4);
        public static PowerMonomialBuilder X5 => Factor(5);
        public static PowerMonomialBuilder X6 => Factor(6);
        public static PowerMonomialBuilder X7 => Factor(7);
        public static PowerMonomialBuilder X8 => Factor(8);
        public static PowerMonomialBuilder X9 => Factor(9);
    }
}