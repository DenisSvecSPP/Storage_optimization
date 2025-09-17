namespace Cmdty.Storage
{
    public class InterpolationType
    {

        private InterpolationType()
        {
        }

        public static InterpolationType PiecewiseLinear => new PiecewiseLinearType();
        public static InterpolationType Polynomial => new PolynomialType();
        public static InterpolationType PolynomialWithParams(double newtonRaphsonAccuracy = 1E-10, int newtonRaphsonMaxNumIterations = 100,
                        int newtonRaphsonSubdivision = 20)
                    => new PolynomialType(newtonRaphsonAccuracy, newtonRaphsonMaxNumIterations, newtonRaphsonSubdivision);
        public static InterpolationType Step => new StepType();

        public sealed class PiecewiseLinearType : InterpolationType
        {
        }

        public sealed class StepType : InterpolationType
        {
        }


        public sealed class PolynomialType : InterpolationType
        {
            public double NewtonRaphsonAccuracy { get; }
            public int NewtonRaphsonMaxNumIterations { get; }
            public int NewtonRaphsonSubdivision { get; }

            public PolynomialType(double newtonRaphsonAccuracy = 1E-10, int newtonRaphsonMaxNumIterations = 100,
                                    int newtonRaphsonSubdivision = 20)
            {
                NewtonRaphsonAccuracy = newtonRaphsonAccuracy;
                NewtonRaphsonMaxNumIterations = newtonRaphsonMaxNumIterations;
                NewtonRaphsonSubdivision = newtonRaphsonSubdivision;
            }

        }

    }
}