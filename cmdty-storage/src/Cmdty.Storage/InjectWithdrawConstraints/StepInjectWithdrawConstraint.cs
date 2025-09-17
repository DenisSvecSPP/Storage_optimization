#region License
// Copyright (c) 2021 Jake Fowler
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
using JetBrains.Annotations;

namespace Cmdty.Storage
{
    public sealed class StepInjectWithdrawConstraint : IInjectWithdrawConstraint
    {
        private readonly InjectWithdrawRangeByInventory[] _injectWithdrawRanges;
        private readonly double[] _inventories;

        public StepInjectWithdrawConstraint([NotNull] IEnumerable<InjectWithdrawRangeByInventory> injectWithdrawRanges)
        {
            if (injectWithdrawRanges == null) throw new ArgumentNullException(nameof(injectWithdrawRanges));

            _injectWithdrawRanges = injectWithdrawRanges.OrderBy(injectWithdrawRange => injectWithdrawRange.Inventory)
                                                        .ToArray();
            if (_injectWithdrawRanges.Length < 2)
                throw new ArgumentException("Inject/withdraw ranges collection must contain at least two elements.", nameof(injectWithdrawRanges));


            InjectWithdrawRange secondHighestInventoryRange = _injectWithdrawRanges[_injectWithdrawRanges.Length - 2].InjectWithdrawRange;
            InjectWithdrawRange highestInventoryRange = _injectWithdrawRanges[_injectWithdrawRanges.Length - 1].InjectWithdrawRange;
            
            if (!StorageHelper.EqualsWithinTol(secondHighestInventoryRange.MaxInjectWithdrawRate,highestInventoryRange.MaxInjectWithdrawRate, 1E-12))
                throw new ArgumentException("Top two ratchets do not have he same max injection rate.", nameof(injectWithdrawRanges));
            if (!StorageHelper.EqualsWithinTol(secondHighestInventoryRange.MinInjectWithdrawRate, highestInventoryRange.MinInjectWithdrawRate, 1E-12))
                throw new ArgumentException("Top two ratchets do not have he same max withdrawal rate.", nameof(injectWithdrawRanges));
            
            _inventories = _injectWithdrawRanges.Select(injectWithdrawRange => injectWithdrawRange.Inventory)
                                                .ToArray();

            if (_inventories.Length > 2)
            {
                for (int i = 1; i < _inventories.Length - 1; i++) // Don't check the last pair as these should have same inject/withdraw rates, as checked for above
                {
                    if (_injectWithdrawRanges[i].InjectWithdrawRange.MaxInjectWithdrawRate > _injectWithdrawRanges[i-1].InjectWithdrawRange.MaxInjectWithdrawRate)
                        throw new ArgumentException("Ratchet injection rates cannot increase with inventory.");
                    if (_injectWithdrawRanges[i].InjectWithdrawRange.MinInjectWithdrawRate > _injectWithdrawRanges[i - 1].InjectWithdrawRange.MinInjectWithdrawRate)
                        throw new ArgumentException("Ratchet withdrawal rates cannot decrease with inventory.");
                }
            }

        }

        public InjectWithdrawRange GetInjectWithdrawRange(double inventory)
        {
            if (inventory < _inventories[0] || inventory > _inventories[_inventories.Length - 1])
                throw new ArgumentException($"Value of inventory is outside of the interval [{_inventories[0]}, {_inventories[_inventories.Length - 1]}].", nameof(inventory));
            int searchIndex = Array.BinarySearch(_inventories, inventory);
            int index = searchIndex >= 0 ? searchIndex : ~searchIndex - 1; 
            return _injectWithdrawRanges[index].InjectWithdrawRange;
        }

        public double InventorySpaceUpperBound(double nextPeriodInventorySpaceLowerBound, double nextPeriodInventorySpaceUpperBound,
            double currentPeriodMinInventory, double currentPeriodMaxInventory, double inventoryPercentLoss)
        {
            InjectWithdrawRange currentPeriodInjectWithdrawRangeAtMaxInventory = GetInjectWithdrawRange(currentPeriodMaxInventory);

            double nextPeriodMaxInventoryFromThisPeriodMaxInventory = currentPeriodMaxInventory * (1 - inventoryPercentLoss)
                                                                      + currentPeriodInjectWithdrawRangeAtMaxInventory.MaxInjectWithdrawRate;
            double nextPeriodMinInventoryFromThisPeriodMaxInventory = currentPeriodMaxInventory * (1 - inventoryPercentLoss)
                                                                      + currentPeriodInjectWithdrawRangeAtMaxInventory.MinInjectWithdrawRate;

            if (nextPeriodMinInventoryFromThisPeriodMaxInventory <= nextPeriodInventorySpaceUpperBound &&
                nextPeriodInventorySpaceLowerBound <= nextPeriodMaxInventoryFromThisPeriodMaxInventory)
            {
                // No need to solve root as next period inventory space can be reached from the current period max inventory
                return currentPeriodMaxInventory;
            }
            // TODO share code in method up to here with PiecewiseLinearInjectWithdrawConstraint

            double? inventorySpaceUpper = null;
            for (int i = 0; i < _injectWithdrawRanges.Length - 1; i++)
            {
                // TODO reuse values between loop iterations like in PiecewiseLinearInjectWithdrawConstraint, or not bother because will make code less clear?
                double maxWithdrawRate = _injectWithdrawRanges[i].InjectWithdrawRange.MinInjectWithdrawRate;
                double bracketLowerInventory = _injectWithdrawRanges[i].Inventory;
                double bracketLowerInventoryAfterWithdraw = bracketLowerInventory * (1 - inventoryPercentLoss) + maxWithdrawRate;
                
                double bracketUpperInventory = _injectWithdrawRanges[i + 1].Inventory;
                double bracketUpperInventoryAfterWithdraw = bracketUpperInventory * (1 - inventoryPercentLoss) + maxWithdrawRate;

                if (bracketLowerInventoryAfterWithdraw <= nextPeriodInventorySpaceUpperBound &&
                    nextPeriodInventorySpaceUpperBound <= bracketUpperInventoryAfterWithdraw)
                {
                    // If there are multiple solutions we want to take the maximum one, so we keep overwriting the solution
                    inventorySpaceUpper = StorageHelper.InterpolateLinearAndSolve(bracketLowerInventory,
                                        bracketLowerInventoryAfterWithdraw, bracketUpperInventory,
                                        bracketUpperInventoryAfterWithdraw, nextPeriodInventorySpaceUpperBound);
                }
            }

            if (inventorySpaceUpper == null)
                throw new ApplicationException("Storage inventory constraints cannot be satisfied.");
            return inventorySpaceUpper.Value;
        }

        public double InventorySpaceLowerBound(double nextPeriodInventorySpaceLowerBound, double nextPeriodInventorySpaceUpperBound,
            double currentPeriodMinInventory, double currentPeriodMaxInventory, double inventoryPercentLoss)
        {
            InjectWithdrawRange currentPeriodInjectWithdrawRangeAtMinInventory = GetInjectWithdrawRange(currentPeriodMinInventory);

            double nextPeriodMaxInventoryFromThisPeriodMinInventory = currentPeriodMinInventory * (1 - inventoryPercentLoss)
                                                                      + currentPeriodInjectWithdrawRangeAtMinInventory.MaxInjectWithdrawRate;
            double nextPeriodMinInventoryFromThisPeriodMinInventory = currentPeriodMinInventory * (1 - inventoryPercentLoss)
                                                                      + currentPeriodInjectWithdrawRangeAtMinInventory.MinInjectWithdrawRate;

            if (nextPeriodMinInventoryFromThisPeriodMinInventory <= nextPeriodInventorySpaceUpperBound &&
                nextPeriodInventorySpaceLowerBound <= nextPeriodMaxInventoryFromThisPeriodMinInventory)
            {
                // No need to solve root as next period inventory space can be reached from the current period min inventory
                return currentPeriodMinInventory;
            }
            // TODO share code in method up to here with PiecewiseLinearInjectWithdrawConstraint

            double? inventorySpaceLower = null;
            for (int i = _injectWithdrawRanges.Length - 2; i >= 0; i--)
            {
                // TODO reuse values between loop iterations like in PiecewiseLinearInjectWithdrawConstraint, or not bother because will make code less clear?
                double maxInjectionRate = _injectWithdrawRanges[i].InjectWithdrawRange.MaxInjectWithdrawRate;
                double bracketLowerInventory = _injectWithdrawRanges[i].Inventory;
                double bracketLowerInventoryAfterInject = bracketLowerInventory * (1 - inventoryPercentLoss) + maxInjectionRate;
                double bracketUpperInventory = _injectWithdrawRanges[i + 1].Inventory;
                double bracketUpperInventoryAfterInject = bracketUpperInventory * (1 - inventoryPercentLoss) + maxInjectionRate;

                if (bracketLowerInventoryAfterInject <= nextPeriodInventorySpaceLowerBound &&
                    nextPeriodInventorySpaceLowerBound <= bracketUpperInventoryAfterInject)
                {
                    // If there are multiple solutions we want to take the minimum one, so we keep overwriting the solution
                    inventorySpaceLower = StorageHelper.InterpolateLinearAndSolve(bracketLowerInventory,
                        bracketLowerInventoryAfterInject, bracketUpperInventory,
                        bracketUpperInventoryAfterInject, nextPeriodInventorySpaceLowerBound);
                }
            }

            if (inventorySpaceLower == null)
                throw new ApplicationException("Storage inventory constraints cannot be satisfied.");
            return inventorySpaceLower.Value;
        }
    }
}
