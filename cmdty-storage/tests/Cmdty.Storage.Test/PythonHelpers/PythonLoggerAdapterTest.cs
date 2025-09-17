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
using Cmdty.Storage.PythonHelpers;
using Microsoft.Extensions.Logging;
using Xunit;

namespace Cmdty.Storage.Test
{
    public sealed class PythonLoggerAdapterTest
    {

        [Fact]
        [Trait("Category", "PythonHelpers")]
        public void LogInformation_AsExpected()
        {
            (List<int> isEnabledCalls, List<(int logLevel, string message)> logCalls, PythonLoggerAdapter<LsmcStorageValuation> logAdapter) 
                    = CreateLogAdapter();

            logAdapter.LogInformation("Hello {0}, {1}.", "one", "two");
            
            Assert.Equal(new int[0], isEnabledCalls);
            Assert.Equal(new (int logLevel, string message)[]{(20, "Hello one, two.")}, logCalls);
        }

        [Fact]
        [Trait("Category", "PythonHelpers")]
        public void IsEnabled_LogLevelError_AsExpected()
        {
            (List<int> isEnabledCalls, List<(int logLevel, string message)> logCalls, PythonLoggerAdapter<LsmcStorageValuation> logAdapter)
                = CreateLogAdapter();

            logAdapter.IsEnabled(LogLevel.Error);

            Assert.Equal(new int[]{40}, isEnabledCalls);
            Assert.Equal(new (int logLevel, string message)[0], logCalls);
        }

        [Fact]
        [Trait("Category", "PythonHelpers")]
        public void LogCritical_WithException_AsExpected()
        {
            (List<int> isEnabledCalls, List<(int logLevel, string message)> logCalls, PythonLoggerAdapter<LsmcStorageValuation> logAdapter)
                = CreateLogAdapter();

            var exception = new ApplicationException("Some error message.");
            logAdapter.LogCritical(exception,"Error {0}, {1}.", "one", "two");

            Assert.Equal(new int[0], isEnabledCalls);
            Assert.Equal(new (int logLevel, string message)[] { (50, $"Error one, two.{Environment.NewLine}{exception}") }, logCalls);
        }

        private static (List<int> isEnabledCalls, List<(int logLevel, string message)> logCalls, PythonLoggerAdapter<LsmcStorageValuation> logAdapter) CreateLogAdapter()
        {
            var isEnabledCalls = new List<int>();
            var logCalls = new List<(int logLevel, string message)>();

            var logAdapter = new PythonLoggerAdapter<LsmcStorageValuation>(logLevel =>
            {
                isEnabledCalls.Add(logLevel);
                return true;
            }, (logLevel, message) => { logCalls.Add((logLevel, message)); });
            return (isEnabledCalls, logCalls, logAdapter);
        }
        
    }
}
