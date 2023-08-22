Promote ``qupulse.expression`` to a subpackage and create ``qupulse.expression.protocol`` with protocol classes that define the expression interface that is supposed to be used by qupulse.
The ```sympy`` based implementation is moved to ``qupulse.expressions.sympy`` and imported in ``qupulse.expressions``.

The intended use is to be able to use less powerful but faster implementations of the ``Expression`` protocol where appropriate.
In this first iteration, qupulse still relies on internals of the ``sympy`` based implementation in many places which is to be removed in the future.
