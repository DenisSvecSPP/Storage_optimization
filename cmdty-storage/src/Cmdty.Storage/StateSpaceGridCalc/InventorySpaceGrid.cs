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
using System.Linq;
using Cmdty.TimePeriodValueTypes;

namespace Cmdty.Storage
{
    public static class InventorySpaceGrid
    {
        // TODO delete this and just use FixedSpacingStateSpaceGridCalc.CreateForFixedNumberOfPointsOnGlobalInventoryRange
        public static Func<ICmdtyStorage<T>, IDoubleStateSpaceGridCalc> FixedNumberOfPointsOnGlobalInventoryRangeFactory<T>(int numGridPointsOverGlobalInventoryRange)
            where T : ITimePeriod<T>
        {
            if (numGridPointsOverGlobalInventoryRange < 3)
                throw new ArgumentException($"Parameter {nameof(numGridPointsOverGlobalInventoryRange)} value must be at least 3.", nameof(numGridPointsOverGlobalInventoryRange));

            IDoubleStateSpaceGridCalc GridCalcFactory(ICmdtyStorage<T> storage)
            {
                T[] storagePeriods = storage.StartPeriod.EnumerateTo(storage.EndPeriod).ToArray();

                double globalMaxInventory = storagePeriods.Max(storage.MaxInventory);
                double globalMinInventory = storagePeriods.Min(storage.MinInventory);
                double gridSpacing = (globalMaxInventory - globalMinInventory) /
                                     (numGridPointsOverGlobalInventoryRange - 1);
                return new FixedSpacingStateSpaceGridCalc(gridSpacing);
            }

            return GridCalcFactory;
        }


    }
}
