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
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Cmdty.Core.Simulation.MultiFactor;
using Cmdty.TimePeriodValueTypes;
using Cmdty.TimeSeries;
using ExcelDna.Integration;

namespace Cmdty.Storage.Excel
{
    //public class MultiFactorCalcWrapper : ExcelCalcWrapper
    //{
    //    public MultiFactorCalcWrapper()
    //    {
    //        CancellationToken cancelToken = _cancellationTokenSource.Token;
    //        Status = CalcStatus.Running;
    //        CalcTask = Task.Run(() =>
    //        {
    //            TimeSpan timeSpan = TimeSpan.FromSeconds(1);
    //            UpdateProgress(0.0);
    //            cancelToken.ThrowIfCancellationRequested();
    //            Thread.Sleep(timeSpan);
    //            UpdateProgress(0.2);
    //            cancelToken.ThrowIfCancellationRequested();
    //            Thread.Sleep(timeSpan);
    //            UpdateProgress(0.4);
    //            cancelToken.ThrowIfCancellationRequested();
    //            Thread.Sleep(timeSpan);
    //            UpdateProgress(0.6);
    //            cancelToken.ThrowIfCancellationRequested();
    //            Thread.Sleep(timeSpan);
    //            UpdateProgress(0.8);
    //            //throw new Exception("My exception");
    //            cancelToken.ThrowIfCancellationRequested();
    //            Thread.Sleep(timeSpan);
    //            UpdateProgress(0.9);
    //            cancelToken.ThrowIfCancellationRequested();
    //            Thread.Sleep(timeSpan);
    //            cancelToken.ThrowIfCancellationRequested();
    //            UpdateProgress(1.0);
    //            return (object)12345.6789;
    //        }, cancelToken);

    //        CalcTask.ContinueWith(UpdateStatus);
            
    //    }

    //    public new bool CancellationSupported() => true;

    //}

    public static class MultiFactorXl
    {
        private static readonly Dictionary<string, ExcelCalcWrapper> _calcWrappers = new Dictionary<string, ExcelCalcWrapper>();
        private static readonly Dictionary<string, CmdtyStorage<Day>> _storageObjects = new Dictionary<string, CmdtyStorage<Day>>();

        [ExcelFunction(Name = AddIn.ExcelFunctionNamePrefix + nameof(CreateStorage),
            Description = "Creates and caches an object representing a storage facility.",
            Category = AddIn.ExcelFunctionCategory, IsThreadSafe = false, IsVolatile = false, IsExceptionSafe = true)] // TODO turn IsThreadSafe to true and use ConcurrentDictionary?
        public static object CreateStorage(
            [ExcelArgument(Name = "Storage_name", Description = "Name of storage object to create.")] string name,
            [ExcelArgument(Name = ExcelArg.StorageStart.Name, Description = ExcelArg.StorageStart.Description)] DateTime storageStart,
            [ExcelArgument(Name = ExcelArg.StorageEnd.Name, Description = ExcelArg.StorageEnd.Description)] DateTime storageEnd,
            [ExcelArgument(Name = ExcelArg.Ratchets.Name, Description = ExcelArg.Ratchets.Description)] object ratchets,
            [ExcelArgument(Name = ExcelArg.RatchetInterpolation.Name, Description = ExcelArg.RatchetInterpolation.Description)] string ratchetInterpolation,
            [ExcelArgument(Name = ExcelArg.InjectionCost.Name, Description = ExcelArg.InjectionCost.Description)] double injectionCostRate,
            [ExcelArgument(Name = ExcelArg.CmdtyConsumedInject.Name, Description = ExcelArg.CmdtyConsumedInject.Description)] double cmdtyConsumedOnInjection,
            [ExcelArgument(Name = ExcelArg.WithdrawalCost.Name, Description = ExcelArg.WithdrawalCost.Description)] double withdrawalCostRate,
            [ExcelArgument(Name = ExcelArg.CmdtyConsumedWithdraw.Name, Description = ExcelArg.CmdtyConsumedWithdraw.Description)] double cmdtyConsumedOnWithdrawal,
            [ExcelArgument(Name = ExcelArg.NumericalTolerance.Name, Description = ExcelArg.NumericalTolerance.Description)] object numericalToleranceIn)
        {
            return StorageExcelHelper.ExecuteExcelFunction(() =>
            {
                double numericalTolerance = StorageExcelHelper.DefaultIfExcelEmptyOrMissing(numericalToleranceIn, 1E-10, "Numerical_tolerance");
                CmdtyStorage<Day> storage = StorageExcelHelper.CreateCmdtyStorageFromExcelInputs<Day>(storageStart,
                    storageEnd, ratchets, ratchetInterpolation, injectionCostRate, cmdtyConsumedOnInjection,
                    withdrawalCostRate, cmdtyConsumedOnWithdrawal, numericalTolerance);
                _storageObjects[name] = storage;
                return name;
            });
        }


        [ExcelFunction(Name = AddIn.ExcelFunctionNamePrefix + nameof(StorageValueThreeFactor),
            Description = "Calculates the NPV, Deltas, Trigger prices and other metadata using a 3-factor seasonal model of price dynamics.",
            Category = AddIn.ExcelFunctionCategory, IsThreadSafe = false, IsVolatile = false, IsExceptionSafe = true)] // TODO turn IsThreadSafe to true and use ConcurrentDictionary?
        public static object StorageValueThreeFactor(
            [ExcelArgument(Name = "Name", Description = "Name of cached object to create.")] string name,
            [ExcelArgument(Name = ExcelArg.StorageHandle.Name, Description = ExcelArg.StorageHandle.Description)] string storageHandle,
            [ExcelArgument(Name = ExcelArg.ValDate.Name, Description = ExcelArg.ValDate.Description)] DateTime valuationDate,
            [ExcelArgument(Name = ExcelArg.Inventory.Name, Description = ExcelArg.Inventory.Description)] double currentInventory,
            [ExcelArgument(Name = ExcelArg.ForwardCurve.Name, Description = ExcelArg.ForwardCurve.Description)] object forwardCurve,
            [ExcelArgument(Name = ExcelArg.InterestRateCurve.Name, Description = ExcelArg.InterestRateCurve.Description)] object interestRateCurve,
            [ExcelArgument(Name = ExcelArg.SpotVol.Name, Description = ExcelArg.SpotVol.Description)] double spotVol,
            [ExcelArgument(Name = ExcelArg.SpotMeanReversion.Name, Description = ExcelArg.SpotMeanReversion.Description)] double spotMeanReversion,
            [ExcelArgument(Name = ExcelArg.LongTermVol.Name, Description = ExcelArg.LongTermVol.Description)] double longTermVol,
            [ExcelArgument(Name = ExcelArg.SeasonalVol.Name, Description = ExcelArg.SeasonalVol.Description)] double seasonalVol,
            [ExcelArgument(Name = ExcelArg.DiscountDeltas.Name, Description = ExcelArg.DiscountDeltas.Description)] bool discountDeltas,
            [ExcelArgument(Name = ExcelArg.SettleDates.Name, Description = ExcelArg.SettleDates.Description)] object settleDatesIn,
            [ExcelArgument(Name = ExcelArg.NumSims.Name, Description = ExcelArg.NumSims.Description)] int numSims,
            [ExcelArgument(Name = ExcelArg.BasisFunctions.Name, Description = ExcelArg.BasisFunctions.Description)] string basisFunctionsIn,
            [ExcelArgument(Name = ExcelArg.Seed.Name, Description = ExcelArg.Seed.Description)] object seedIn,
            [ExcelArgument(Name = ExcelArg.ForwardSimSeed.Name, Description = ExcelArg.ForwardSimSeed.Description)] object fwdSimSeedIn,
            [ExcelArgument(Name = ExcelArg.NumGridPoints.Name, Description = ExcelArg.NumGridPoints.Description)] object numGlobalGridPointsIn,
            [ExcelArgument(Name = ExcelArg.NumericalTolerance.Name, Description = ExcelArg.NumericalTolerance.Description)] object numericalTolerance,
            [ExcelArgument(Name = ExcelArg.ExtraDecisions.Name, Description = ExcelArg.ExtraDecisions.Description)] object extraDecisions)
        {
            return StorageExcelHelper.ExecuteExcelFunction(() =>
            {
                _calcWrappers[name] = ExcelCalcWrapper.CreateCancellable((cancellationToken, onProgress) =>
                {
                    // TODO provide alternative method for interpolating interest rates
                    Func<Day, double> interpolatedInterestRates =
                        StorageExcelHelper.CreateLinearInterpolatedInterestRateFunc(interestRateCurve, ExcelArg.InterestRateCurve.Name);

                    Func<Day, Day, double> discountFunc = StorageHelper.CreateAct65ContCompDiscounter(interpolatedInterestRates);
                    Day valDate = Day.FromDateTime(valuationDate);
                    Func<Day, Day> settleDateRule = StorageExcelHelper.CreateSettlementRule(settleDatesIn, ExcelArg.SettleDates.Name);

                    CmdtyStorage<Day> storage = _storageObjects[storageHandle];
                    int numGlobalGridPoints = StorageExcelHelper.DefaultIfExcelEmptyOrMissing(numGlobalGridPointsIn, ExcelArg.NumGridPoints.Default,
                                                                    ExcelArg.NumGridPoints.Name);
                    
                    string basisFunctionsText = basisFunctionsIn.Replace("x_st", "x0").Replace("x_lt", "x1").Replace("x_sw", "x2");
                    
                    var lsmcParamsBuilder = new LsmcValuationParameters<Day>.Builder
                    {
                        Storage = storage, 
                        CurrentPeriod = valDate, 
                        Inventory = currentInventory,
                        ForwardCurve = StorageExcelHelper.CreateDoubleTimeSeries<Day>(forwardCurve, ExcelArg.ForwardCurve.Name),
                        DiscountFactors = discountFunc,
                        
                        DiscountDeltas = discountDeltas,
                        BasisFunctions = BasisFunctionsBuilder.Parse(basisFunctionsText),

                        ExtraDecisions = StorageExcelHelper.DefaultIfExcelEmptyOrMissing(extraDecisions, 0, ExcelArg.ExtraDecisions.Name),
                        CancellationToken = cancellationToken,
                        OnProgressUpdate = onProgress,
                        GridCalc = FixedSpacingStateSpaceGridCalc.CreateForFixedNumberOfPointsOnGlobalInventoryRange(storage, numGlobalGridPoints),
                        NumericalTolerance = StorageExcelHelper.DefaultIfExcelEmptyOrMissing(numericalTolerance, LsmcValuationParameters<Day>.Builder.DefaultNumericalTolerance, ExcelArg.NumericalTolerance.Description),
                        SettleDateRule = settleDateRule
                    };

                    // TODO test that this works with expired storage
                    Day endDate = new[] {valDate, storage.EndPeriod}.Max();
                    var threeFactorParams =
                        MultiFactorParameters.For3FactorSeasonal(spotMeanReversion, spotVol, longTermVol, seasonalVol, valDate, endDate);

                    // TODO better error messages if seedIn and fwdSimSeedIn cannot be cast
                    int? seed = StorageExcelHelper.IsExcelEmptyOrMissing(seedIn) ? (int?)null : (int)(double)seedIn;
                    int? fwdSimSeed = StorageExcelHelper.IsExcelEmptyOrMissing(fwdSimSeedIn) ? (int?)null : (int)(double)fwdSimSeedIn;
                    
                    lsmcParamsBuilder.SimulateWithMultiFactorModelAndMersenneTwister(threeFactorParams, numSims,seed, fwdSimSeed);

                    return LsmcStorageValuation.WithNoLogger.Calculate(lsmcParamsBuilder.Build());
                });
                return name;
            });
        }

        [ExcelFunction(Name = AddIn.ExcelFunctionNamePrefix + nameof(SubscribeProgress),
            Description = "TODO.", // TODO
            Category = AddIn.ExcelFunctionCategory, IsThreadSafe = false, IsVolatile = false, IsExceptionSafe = true, IsClusterSafe =true)] // TODO turn IsThreadSafe to true and use ConcurrentDictionary?
        public static object SubscribeProgress(string name)
        {
            return StorageExcelHelper.ExecuteExcelFunction(() =>
            {
                const string functionName = nameof(SubscribeProgress);
                return ExcelAsyncUtil.Observe(functionName, name, () =>
                {
                    ExcelCalcWrapper wrapper = _calcWrappers[name];
                    var excelObserver = new CalcWrapperProgressObservable(wrapper);
                    return excelObserver;
                });
            });
        }

        [ExcelFunction(Name = AddIn.ExcelFunctionNamePrefix + nameof(SubscribeStatus),
            Description = "TODO.", // TODO
            Category = AddIn.ExcelFunctionCategory, IsThreadSafe = false, IsVolatile = false, IsExceptionSafe = true)] // TODO turn IsThreadSafe to true and use ConcurrentDictionary?
        public static object SubscribeStatus(string name)
        {
            return StorageExcelHelper.ExecuteExcelFunction(() =>
            {
                const string functionName = nameof(SubscribeStatus);
                return ExcelAsyncUtil.Observe(functionName, name, () =>
                {
                    ExcelCalcWrapper wrapper = _calcWrappers[name];
                    var excelObserver = new CalcWrapperStatusObservable(wrapper);
                    return excelObserver;
                });
            });
        }

        [ExcelFunction(Name = AddIn.ExcelFunctionNamePrefix + nameof(SubscribeResultProperty),
            Description = "TODO.", // TODO
            Category = AddIn.ExcelFunctionCategory, IsThreadSafe = false, IsVolatile = false, IsExceptionSafe = true)] // TODO turn IsThreadSafe to true and use ConcurrentDictionary?
        public static object SubscribeResultProperty(string objectHandle, string propertyName, object returnedWhilstWaiting)
        {
            return StorageExcelHelper.ExecuteExcelFunction(() =>
            {
                const string functionName = nameof(SubscribeResultProperty);
                return ExcelAsyncUtil.Observe(functionName, new [] { objectHandle, propertyName, returnedWhilstWaiting}, () =>
                {
                    ExcelCalcWrapper wrapper = _calcWrappers[objectHandle];
                    var excelObservable = new CalcWrapperResultPropertyObservable(wrapper, propertyName, returnedWhilstWaiting);
                    return excelObservable;
                });
            });
        }

        //[ExcelFunction(Name = AddIn.ExcelFunctionNamePrefix + nameof(GetError),
        //    Description = "TODO.", // TODO
        //    Category = AddIn.ExcelFunctionCategory, IsThreadSafe = false, IsVolatile = false, IsExceptionSafe = true)]
        //public static object GetError()
        //{
        //    return ExcelError.ExcelErrorGettingData;
        //}

    }

    sealed class CalcWrapperResultPropertyObservable : CalcWrapperObservableBase
    {
        private readonly object _returnedWhilstWaiting;
        private static readonly object[] GetterParams = new object[0];
        private readonly MethodInfo _propertyGetter;
        
        public CalcWrapperResultPropertyObservable(ExcelCalcWrapper calcWrapper, string resultPropertyName, object returnedWhilstWaiting) : base(calcWrapper)
        {
            PropertyInfo[] properties = calcWrapper.ResultType.GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.GetProperty);
            PropertyInfo propertyInfo = properties.FirstOrDefault(info => info.Name.Equals(resultPropertyName, StringComparison.OrdinalIgnoreCase));

            if (propertyInfo == null)
                throw new ArgumentException($"Result type {calcWrapper.ResultType.Name} has not public instance property called {resultPropertyName}."); // TODO test and maybe rework
            _propertyGetter = propertyInfo.GetMethod;

            _returnedWhilstWaiting = returnedWhilstWaiting;
            calcWrapper.CalcTask.ContinueWith(task =>
                {
                    if (task.Status == TaskStatus.RanToCompletion)
                        PropertyValueUpdate();
                });
        }

        private static object GetPropertyValueToReturn(MethodInfo propertyGetter, object resultObject)
        {
            object propertyValue = propertyGetter.Invoke(resultObject, GetterParams);
            return StorageExcelHelper.TransformForExcelReturn(propertyValue);
        }

        protected override void OnSubscribe()
        {
            if (_calcWrapper.Status == CalcStatus.Success) // Has already completed
                PropertyValueUpdate();
            else
                _observer?.OnNext(_returnedWhilstWaiting);
        }

        private void PropertyValueUpdate()
        {
            object propertyValue = GetPropertyValueToReturn(_propertyGetter, _calcWrapper.CalcTask.Result); 
            _observer?.OnNext(propertyValue);
        }

    }

    sealed class CalcWrapperStatusObservable : CalcWrapperObservableBase
    {
        public CalcWrapperStatusObservable(ExcelCalcWrapper calcWrapper) : base(calcWrapper)    
            => calcWrapper.CalcTask.ContinueWith(task => TaskStatusUpdate(_calcWrapper.Status));
        
        private void TaskStatusUpdate(CalcStatus calcStatus) => _observer?.OnNext(calcStatus.ToString("G"));
        
        protected override void OnSubscribe() => TaskStatusUpdate(_calcWrapper.Status); // TODO could be synchronizations issues, might need to lock
    }

    sealed class CalcWrapperProgressObservable : CalcWrapperObservableBase
    {
        public CalcWrapperProgressObservable(ExcelCalcWrapper calcWrapper) : base(calcWrapper)
                    => _calcWrapper.OnProgressUpdate += ProgressUpdate;
        
        internal void ProgressUpdate(double progress) => _observer?.OnNext(progress);

        protected override void OnSubscribe() => _observer.OnNext(_calcWrapper.Progress);

        protected override void OnDispose()
        {
            _calcWrapper.OnProgressUpdate -= ProgressUpdate;
            base.OnDispose();
        }
    }

    abstract class CalcWrapperObservableBase : IExcelObservable, IDisposable
    {
        protected readonly ExcelCalcWrapper _calcWrapper;
        protected IExcelObserver _observer;

        protected CalcWrapperObservableBase(ExcelCalcWrapper calcWrapper)
        {
            _calcWrapper = calcWrapper;
            _calcWrapper.CalcTask.ContinueWith(task =>
            {
                if (task.IsFaulted) // TODO what about cancelled
                    _observer?.OnError(task.Exception.InnerException);
                else 
                    _observer?.OnCompleted();
            });
        }

        public IDisposable Subscribe(IExcelObserver excelObserver)
        {
            _observer = excelObserver;
            OnSubscribe();
            return this;
        }

        protected abstract void OnSubscribe();
        
        protected virtual void OnDispose() => _observer = null;

        public void Dispose() => OnDispose();

    }

}
