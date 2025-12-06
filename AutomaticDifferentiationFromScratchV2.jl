### A Pluto.jl notebook ###
# v0.20.21

#> [frontmatter]
#> title = "Automatic Differentiation from Scratch"
#> description = "Naive but educational implementation of automatic differentiation."
#> 
#>     [[frontmatter.author]]
#>     name = "Andreas Weh"

using Markdown
using InteractiveUtils

# ╔═╡ b8ca321b-736c-4a29-bec6-041bdb4c28fc
using Symbolics

# ╔═╡ 6cd9539e-c4e0-4333-af8f-292e47cc539b
using PlutoUI

# ╔═╡ de19ba50-af83-11f0-8445-075b7cd7e011
md"""
# Automatic differentiation from scratch

Derivatives are ubiquitous in physics, take stress, strain, heat capacity, expansion coefficients, or response functions in general.
Furthermore, they are a useful tool for optimization problems.

This notebook shows a naive minimal implementation of automatic differentiation in Julia.
But let's first start out with the traditional approaches and their limitations.
"""

# ╔═╡ 1d9e9278-4dcd-4a22-a3ff-2a0d5426e4c1
md"""
# Finite differences

Let's start with the school book definition of a derivative:

```math
	\frac{\partial f(x)}{\partial x}
	= \lim_{h \rightarrow \infty} \frac{f(x+h) - f(x)}{h}
```

The numerical evaluation seems straight forward: just use a small value ``h``.
This is called *finite difference*, as we use a finite value of ``h``.

We use the symmetric central difference

```math
	\frac{\partial f(x)}{\partial x}
	\approx  \frac{f(x+\frac{h}{2}) - f(x-\frac{h}{2})}{h}:
```
"""

# ╔═╡ 27cb8eac-a052-45b6-9b4e-55ee0dd72048
md"""
In general, finite differences have problems with numerical inaccuracies and instabilities, especially when calculating higher order derivatives.
Finite difference become slow, when calculating gradients with respect to many inputs.

Some packages offer to calculate numerical derivates using integration employing the *residue theorem* instead, to avoid numerical inaccuracies, see e.g. [mpmath's `method='quad'`](https://www.mpmath.org/doc/current/calculus/differentiation.html#numerical-derivatives-diff-diffs).
While calculating analytical integrations is hard and differentiation is simple,
numerically it's the other way around: integrating is quite simple while differentiations becomes hard.
"""

# ╔═╡ 2eb2e25c-df4b-4059-8395-c1fab6400bdf
md"""
# Symbolic differentiation

Symbolic differentiation calculates the analytic derivative of functions using symbolic computation.
This is like the pen-and-paper version of doing derivatives.

We'll use the [Symbolics.jl](https://docs.sciml.ai/Symbolics/stable/) computer algebra system.
"""

# ╔═╡ 03ddb110-d167-427a-9694-06dd09a59735
md"""
To perform symbolic calculations, we have to define symbolic variables first:
"""

# ╔═╡ 4dce8bce-42b1-4637-a486-f7c6692664bf
@variables α β c d

# ╔═╡ dab41bab-f9f1-4b04-bcae-f2d1a8d6bcc0
D = Differential(α)

# ╔═╡ 8787daf9-52a1-469f-bd9d-aba957c67318
md"""
The result of symbolic differentiation becomes unwieldy quite fast which can become inefficient.
Especially as soon as branches and loops get involved, symbolic differentiation might reach its limits.
"""

# ╔═╡ 2c41d052-6f0a-40af-be12-6f193b24b971
md"""
# Automatic differentiation

Automatic differentiation is the modern solution: We evaluate the derivative along with the function without building the function graph.

To implement this we define dual numbers, keeping track of the derivative along with the current value.
"""

# ╔═╡ 7aaeb842-b7bd-4081-8a46-db7b0f3459b0
md"""
## The most naive way
"""

# ╔═╡ cac73c9a-0a18-4c1d-87eb-b54bb991415b
begin
	struct Dual{T} <: Number  # dual numbers are numbers
		x::T
		dx::T
	end
    # constructors for promotion
	Dual(x) = Dual(x, zero(x))
	Dual{T}(x) where T = Dual{T}(x, zero(x))
end

# ╔═╡ 5a590e7e-7837-4a37-99bd-c56370fc41f2
md"""
To teach Julia how to effectively use the new type, we have to define how it interacts with other numbers.
For this purpose, we define a [`promotion_rule`](https://docs.julialang.org/en/v1/manual/conversion-and-promotion/):
"""

# ╔═╡ 01c7ee06-b5ed-48aa-838d-1e2635edae45
begin
	import Base: promote_rule
	# promote regular numbers to dual numbers
	promote_rule(::Type{Dual{T}}, ::Type{T}) where {T<:Number} = Dual{T}
	promote_rule(::Type{Dual{T1}}, ::Type{T2}) where {T1, T2<:Number} =
		Dual{promote_rule(T1, T2)}
	# mostly, Julia figures out symmetry on its own, but we have some edge cases
	promote_rule(::Type{T}, ::Type{Dual{T}}) where {T<:Number} = Dual{T}
	promote_rule(::Type{T1}, ::Type{Dual{T2}}) where {T1<:Number, T2} =
		Dual{promote_rule(T1, T2)}
	# try to promote dual number of different numbers
	promote_rule(::Type{Dual{T1}}, ::Type{Dual{T2}}) where {T1<:Number, T2<:Number} =
		Dual{promote_rule(T1, T2)}
	# extensions for symbols, never do this in packages, this is type piracy!!!
	# type piracy = extending functions we do not own by types we do not own
	promote_rule(::Type{Num}, ::Type{<:Number}) = Num
end

# ╔═╡ 162800f3-c354-4ea6-8259-aabe1605178c
md"""
Other numbers are simply converted to `Dual` numbers of appropriate type.
The default constructor `Dual(x)` is defined such that regular numbers get `dx = 0`.
"""

# ╔═╡ 81ed17a9-d183-47b0-a4c7-593346ebf05e
derivative(x::Dual) = x.dx

# ╔═╡ 4130c505-edb5-41af-bcce-5fa8181763a3
md"""
Now we can define a variable ``x`` and get its derivative with respect to `x`:
"""

# ╔═╡ 42f832ad-82b3-4076-ae69-6bf14ead9038
x = Dual(3.0, 1.0)

# ╔═╡ 24b9af40-f2c0-48f1-a442-24a7dfdd1ebf
derivative(x)

# ╔═╡ 0e264270-8dff-4359-bb37-be5578bd7c33
md"""
If we have a different variable, say ``y``, we set the derivative to 0, as ``\partial y / \partial x = 0``. 
"""

# ╔═╡ a636e3e2-9f63-48ce-94dd-3006bafab90e
y = Dual(7.0, 0.0)

# ╔═╡ 01eb43eb-90bf-4b9f-9f77-9dc2015be8b0
derivative(y)

# ╔═╡ f12b46af-0d0c-458a-897d-66eeab419ece
md"""
So far, we haven't done anything impressive.
We only calculated a trivial derivative, and we even had to specify it ourselves.
So far we cannot do anything useful.
"""

# ╔═╡ 854e677c-bd3f-4e0f-9e07-6818a8713411
md"""
We have to tell Julia how to do math with dual numbers.
So let's tell it what an addition is.
In addition to adding numbers, we have to implement the differentiation rules.
Differentiation is linear

```math
	\frac{\partial(af + bg)}{\partial x} 
	= a \frac{\partial f}{\partial x} + b \frac{\partial g}{\partial x}.
```

Our definition of dual numbers brings already the termination condition for the chain rule to the table:

```math
	\frac{\partial f(x)}{\partial x}
	= f'(x) \frac{\partial x}{\partial x} = f'(x) \cdot 1.
```
We explicitly passed in the ``1``.
So let's add the method for dual numbers to the `+` function:
"""

# ╔═╡ 9cf8876a-0feb-4ace-92f0-4a5ee9753729
Base.:+(a::Dual, b::Dual) = Dual(a.x + b.x, a.dx + b.dx)

# ╔═╡ bea2fffb-e2d6-4f34-8a0a-9d632d5a12bd
md"""
Now that we told Julia how to add dual numbers, we can get derivatives of slightly more complex expressions:
"""

# ╔═╡ c3c7fd11-7f6e-48ad-bbbd-822bf3665248
derivative(x + x)

# ╔═╡ b1598942-2300-4a3c-bb08-1f6dadcbd4c6
derivative(x + y)

# ╔═╡ 97206602-e5fe-4c7e-bbbc-1a68daa46dfc
derivative(x + 3)

# ╔═╡ 58232802-7f9e-4566-96af-14834f8ab885
md"""
Of course, we need the rest of the mathematical operations to do meaningful calculus.
So let's continue with the product rule:

```math
	\frac{\partial (fg)}{\partial x}
	= \frac{\partial f}{\partial x} g + f \frac{\partial g}{\partial x}
```
"""

# ╔═╡ 61826d69-d8c0-4861-9239-a474a49f18f2
Base.:*(a::Dual, b::Dual) = Dual(a.x * b.x, a.dx*b.x + a.x*b.dx)

# ╔═╡ 8c01d6c6-3b89-4539-a78f-9965de105c58
md"""
This is a good point to remember how magic generic programming can be.
Our definition of `Dual` was generic with respect to the data type, so far we only used `Float64`, but it works just es well for other precisions like `Float32`, arbitrary precision (`BigFLoat`), and even symbols!
By implementing automatic differentiation, we get symbolic differentiation for free!

This is quite useful to validate our implementation.
Computers are good with numbers, me, not so much.
"""

# ╔═╡ 58c510a6-a976-454b-98d5-f56f1b0e0b6a
a = Dual(α, Num(1))

# ╔═╡ 8eb80e66-2a15-466c-87ed-4605469a35db
b = Dual(β, Num(0))

# ╔═╡ 9d2cd682-119f-47a4-8af5-b74433b7dca2
md"""
Symbolics.jl defines a specific method for `*` which circumvents our `promote_rule`:
"""

# ╔═╡ 8aa901d5-d29a-48b3-934d-492222834988
# ╠═╡ disabled = true
#=╠═╡
# this depends on execution order...
# So after defining the method, it will show our more specific rule.
@which  c*a  
  ╠═╡ =#

# ╔═╡ ed8ec309-c5df-47b1-8172-a3040af93cd8
md"""
Thus, we have to define a more specific method, to get our multiplication yielding dual numers:
"""

# ╔═╡ e3a7decc-848d-4581-9a62-94a003b3af9b
begin
	import Base: *
	*(a::Dual, b::Num) = a*Dual(b)
	*(a::Num, b::Dual) = *(b, a)
end

# ╔═╡ 6d2f263a-5d86-41f8-8d7c-2cc761cb0555
derivative(2x)

# ╔═╡ 390f48c9-5f46-4a2e-8796-942f2cc7ccc8
derivative(x*y)

# ╔═╡ 34789604-2744-4f19-aced-aa7e6e74d689
derivative(13x+3y)

# ╔═╡ 46fa536a-635e-48e7-9401-d3cf067a5481
which(*, (Number, Num))

# ╔═╡ 7323f5ea-2873-4fe9-9c41-8ad095af807d
md"""
While `Dual isa Number` the specification `Dual` is more precise than just number.
As our method `*(::Num, ::Dual)` is more specific than `Symbolics`'s `*(::Num, ::Number)` it takes precedence.
"""

# ╔═╡ a62415df-7864-4b1b-859a-51c763fdbbfd
derivative(c*a + d*b)

# ╔═╡ 0b1cee5d-3cdd-41b2-bd12-206759d07dc1
derivative(c*a*b)

# ╔═╡ 4be13c18-144e-4cc9-a84e-35c9f2b247af
derivative(a*a)

# ╔═╡ 2b005fd0-fa15-490a-8e3c-d8bbc38fce27
md"""
Let's get the rest of the algebra we need:
"""

# ╔═╡ e17d8292-1394-4d14-9d83-0cbb1dc02ce8
Base.:-(a::Dual, b::Dual) = Dual(a.x - b.x, a.dx - b.dx)

# ╔═╡ 84812014-4448-4639-a321-05f97798a860
begin
	import Base: -
	-(a::Dual, b::Num) = a + Dual(-b)
	-(a::Num, b::Dual) = Dual(a) - b
end

# ╔═╡ 3f0cc04a-d82f-4ac0-b8a5-e4d2a7fe0c3c
# Now we get more tricky derivatives, you might need pen and paper.
Base.:^(a::Dual, b::Dual) = Dual(
	a.x^b.x,
	b.x * a.x^(b.x-1) * a.dx + a.x^b.x * log(a.x) * b.dx
)

# ╔═╡ 0bedb30e-27c2-4552-bb02-9585ac64facf
begin  # again handling Symbolics
	import Base: ^
	^(a::Dual, b::Num) = a^Dual(b)
	^(a::Num, b::Dual) = Dual(a)^b
end

# ╔═╡ f76836b6-dced-420c-b971-6bd1f03f7942
Base.inv(a::Dual) = a^-1.0  # fall back to floating point for simplicity

# ╔═╡ 095aab17-491b-48d9-8124-a92d737bd683
derivative(inv(a))

# ╔═╡ dc9e185a-796c-43ef-9a25-7f2a84d9f69a
Base.:/(a::Dual, b::Dual) = a * b^-1

# ╔═╡ 2c2208ab-b29c-4859-818a-42a7b2f1e62c
derivative(a/b)

# ╔═╡ c1b87c8c-73f3-4cce-8bf1-11028b30f2c2
derivative(b/a)

# ╔═╡ 41408145-200f-4b36-b631-fc6b8bffcce4
derivative(a^7)

# ╔═╡ 47e54957-265c-4964-b1c2-53c55c40195c
derivative(((a + b) / (a - c))^d)

# ╔═╡ f5d16cbe-5daa-4273-9e13-f8ba45f8bf20
md"""
Let's look into our original example of symbolic derivatives.
We need to define geometric functions:
"""

# ╔═╡ 4675c4cf-a6f9-4109-8d00-722a648acb5d
Base.cos(a::Dual) = Dual(cos(a.x), -sin(a.x)*a.dx)

# ╔═╡ f0190ecf-47a0-4618-a8f6-c8a3172a49ad
Base.sin(a::Dual) = Dual(sin(a.x), +cos(a.x)*a.dx)

# ╔═╡ 03cc182f-6cdc-4d40-ae30-10704bcb440c
Base.exp(a::Dual) = Dual(exp(a.x), exp(a.x)*a.dx)

# ╔═╡ d10c042c-0147-4288-9eba-abc4d35341ff
f = exp(α) * sin(α^2 + 3α) / (1 + α^2)^2

# ╔═╡ 54e9a1df-420d-496f-bcd0-eb1448edcce6
df = D(f)

# ╔═╡ e406d151-214e-4f40-9623-29bb636ef871
expand_derivatives(df)

# ╔═╡ 99462628-168a-4cc5-a1a0-ed5d5244d509
f

# ╔═╡ e7f1a6a4-c445-4514-a972-e2fe0e9e1d0f
expand_derivatives(df)

# ╔═╡ 8252153d-2a3c-4615-be65-f23e57dda841
derivative(substitute(f, Dict(α => a)))

# ╔═╡ 4cb1b685-f801-420b-bd70-c1f4c771d0bf
md"""
Or turning the symbolic expression into a function, that we evaluate at our Dual number:
"""

# ╔═╡ 2a4977ab-9749-49f8-a844-3791e015ac7c
derivative(build_function(f, α, expression=Val{false})(a))

# ╔═╡ cac92ee9-ee5d-4ded-8760-cf593f2eb342
md"""
In the above definitions we were being 'smart' about the implementation, reusing other definitions we already provided.
This reduces the chance of making errors, but might be less efficient than a specialized implementation.

So now the big question: what about *flow control*?
Interesting problems require loops and conditionals.

Let's start out with the Θ-step function and the ReLU popular in machine learning:
"""

# ╔═╡ a9a6f108-80a3-4f30-b754-36c4c06a7bd2
md"""
Apparently we still haven't defined all we need.
We forgot about comparisons.
But for comparisons, we have to restrict `Number` to `Real`.
"""

# ╔═╡ 1aba3e89-98e6-42d6-9fdc-1256e99898f7
begin
	# isless is not an arithmetic operation, explicit overloads are necessary
	# But it is enough to implement `<` to get comparisons
	import Base: <
	<(a::Dual{T1}, b::Dual{T2}) where {T1<:Real, T2<:Real} = a.x < b.x
	<(a::Real, b::Dual{<:Real}) = a < b.x
	<(a::Dual{<:Real}, b::Real) = a.x < b
end

# ╔═╡ bbdc708e-293d-41da-b756-c28d03da81fe
Base.zero(a::Dual) = Dual(zero(a.x), zero(a.dx))

# ╔═╡ 707cb394-f408-4088-a76e-57811d12002d
relu(x) = ifelse(x > 0, x, zero(x))

# ╔═╡ b13c0fd9-009e-44be-a328-06ab432a6b33
Base.one(a::Dual) = Dual(one(a.x), zero(a.dx))

# ╔═╡ 3885fa29-592a-4309-9147-9d726078cd9a
step_function(x) = ifelse(x > 0, one(x), zero(x))

# ╔═╡ 36f0e627-d25b-4ab8-8968-b0bf9775a2fa
derivative(step_function(a))

# ╔═╡ afb53e8a-e975-40e2-b663-5b71e9a8e865
md"""
At this point, the magic has left us.
Comparing a symbol (`Num`) to a number does return a boolean but another symbol (`Num`).
Likewise, the `ifelse` for a symbol.

So let's instead go back to evaluating the derivative at numeric values.
"""

# ╔═╡ 2a44be5c-5454-4d3c-9338-321d28f251e6
derivative(step_function(x))

# ╔═╡ 5bce83ec-ce0e-447a-9b4b-d44db39a6493
derivative(relu(x))

# ╔═╡ fc4d4dae-afa4-49d3-bbde-78ec41847001
derivative(relu(y))

# ╔═╡ a6175368-8212-42c5-8e32-a243ae23019d
md"""
Next, let's consider the [arithmetic-geometric mean](https://en.wikipedia.org/wiki/Arithmetic%E2%80%93geometric_mean) (AGM), which is e.g. useful to efficiently evaluate elliptic integrals.
The AGM of `x` and `y` is defined by the limit ``\lim_{n \rightarrow \infty}`` of the series

```math
	a_0 = x, \quad
	g_0 = y, \qquad
	a_{n+1} = \frac{1}{2}(a_n + g_n), \quad
	g_{n+1} = \sqrt{a_n g_n}, \quad
```

where ``a_n`` is the arithmetic mean of the previous results, and ``g_n`` the geometric mean of the previous results.
"""

# ╔═╡ ee3d0cec-08e1-4408-a995-a1a6198e608b
md"""
Let's cross-check our implementation with the [example](https://en.wikipedia.org/wiki/Arithmetic%E2%80%93geometric_mean#Example) from Wikipedia:
"""

# ╔═╡ c788f25b-0f77-4396-a999-9917f99a872f
md"""
We still need to define the absolute.
The derivative is not defined at 0, but we don't really care.
"""

# ╔═╡ 57fdca80-e35f-40bc-8ba0-83d1d777dcbc
Base.abs(a::Dual) = Dual(abs(a.x), a.x*a.dx/abs(a.x))

# ╔═╡ eb13a1cd-7769-4fe2-a8df-e800c34ce7c1
Base.eps(a::Dual) = Dual(eps(a.x), zero(a.dx))  # maybe Dual is not quite meaningful here

# ╔═╡ 74681217-659d-49d8-aa1e-c0c863ea2144
"""Central finite difference."""
function cfd(func, x, h=cbrt(eps(x)))
	return inv(h)*(func(x + h/2) - func(x - h/2))
end

# ╔═╡ 82ca69a8-aa6d-4f90-a912-b2f848e190a3
md"""
Let's check the error using the Taylor approximation

```math
	f(x ± δ) = f(x) ± δ f'(x) + \frac{δ^2}{2}f''(x) + \mathcal{O}(δ^3).
```

Thus, the central finite difference yields
```math
	f(x+\frac{h}{2}) - f(x - \frac{h}{2})
	= h f'(x) + \mathcal{O}(h^3) + ϵ
```
where we added the error ``ϵ`` due to finite precision calculation on the computer.
Thus, we get a numerical error of the order ``h^2 + \frac{ϵ}{h}``.
In general, the error depends on ``f'''(x)``.
Assuming that the function and its third derivative are on a similar scale, the error is minimal for ``h = \sqrt[3]{\frac{ϵ}{2}}``.
As we only use rough numbers, we should not be over-precise and rather drop the factor ``2``.
Thus, a good default choice for the step size ``h`` is
```math
	h^* = \sqrt[3]{ϵ},
```
where ``ϵ`` is the used floating point precision.
For double precision, we get ``h^* = `` $(round(cbrt(eps(Float64)), sigdigits=1));
For double precision, we get ``h^* = `` $(round(cbrt(eps(Float32)), sigdigits=1)).
"""

# ╔═╡ be38cb8c-2825-4738-b490-5e2a89398cc7
Base.sqrt(x::Dual) = x^(1//2)

# ╔═╡ 0a7ea06a-3e3c-400c-8449-dca4e8ae0373
cdf_errors_64 = [
	h => abs(cfd(sqrt, 0.1, h) - inv(2*sqrt(0.1)))
	for h in logrange(1e-2, 1e-10, 9)
]

# ╔═╡ e24c29c0-38f1-48c4-9811-4378fa4adcbe
min_cdf_error_64 = argmin(x -> x.second, cdf_errors_64);

# ╔═╡ d2e065a5-14b1-4558-a7d2-8c3d777c6c5d
md"""
We see that initially the accuracy improves quadratically with the step size:
Reducing ``h`` by one order of magnitude the error decreases by two orders of magnitude.
In our simple toy example, the minimal error is obtained for ``h=``$(min_cdf_error_64.first) at ``\Delta\approx`` $(round(min_cdf_error_64.second, sigdigits=1)).
Decreasing `h` further, round-off errors of finite precision reduce the accuracy.

The problem becomes more apparent when using lower precision:
"""

# ╔═╡ 7c7d4480-de38-4f5e-9a73-e43c77c29881
cdf_errors_32 = [
	h => abs(cfd(sqrt, 0.1f0, h) - inv(2*sqrt(0.1f0)))
	for h in logrange(1f-2, 1f-10, 9)
]

# ╔═╡ d6712129-dcce-4884-8bb2-12724fff2622
abs(cfd(sqrt, 0.1) - inv(2*sqrt(0.1)))

# ╔═╡ e86e42cf-5787-452e-992a-4db9a48f2bd1
abs(cfd(sqrt, 0.1f0) - inv(2*sqrt(0.1f0)))

# ╔═╡ 4fe33ed1-a20c-4840-9ab4-42828e7b8804
"""Algorithmic-geometric mean."""
function agm(x, y)
	a = x
	g = y
	while abs(a - g) > 3*eps(x)
		# @show a, g
		a, g = (1//2)*(a+g), sqrt(a*g)
	end
	return a
end

# ╔═╡ e25b9ee5-bacb-42e1-a947-3dcc8d158368
agm(24.0, 6.0)

# ╔═╡ cca392b5-24c0-4899-af35-972d1ca6777e
agm(x, y)

# ╔═╡ 5de22a47-1386-4b43-b087-224339148166
md"""
## Generalization to multivariate functions
"""

# ╔═╡ 54ca2866-7cea-4c90-978c-e7a1031fc6ec
md"""
So far, we can just differentiate for a single variable.
If we want partial derivatives for multiple variables, we have to reevaluate the functions for each variable.

To generalize the `Dual` numbers to partial derivatives, we can introduce an entry for every variable:
"""

# ╔═╡ 8ccdf71e-3c2f-45fa-afb0-67dadeba12da
struct PartialDual{T, N} <: Number
	x::T
	dx::NTuple{N, T}
end

# ╔═╡ 731769df-3a74-41b3-81a8-1ae226d4bb29
md"""
Here we won't go into further detail and rather refer to [DifferentiationInterface](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/) which provides a common AD interface for many backends, that are usable for real code.
"""

# ╔═╡ 4d436b91-06dc-462b-a2e2-f8704d50d37a
TableOfContents()

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[compat]
PlutoUI = "~0.7.72"
Symbolics = "~6.56.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.2"
manifest_format = "2.0"
project_hash = "69872f4afbe39e52d1981edad17cf0e041d496c3"

[[deps.ADTypes]]
git-tree-sha1 = "27cecae79e5cc9935255f90c53bb831cc3c870d7"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.18.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "7e35fca2bdfba44d797c53dfe63a51fabf39bfc0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.4.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "d81ae5489e13bc03567d4fbbb06c546a5e53c857"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.22.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceMetalExt = "Metal"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Bijections]]
git-tree-sha1 = "a2d308fcd4c2fb90e943cf9cd2fbfa9c32b69733"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.2.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.CompositeTypes]]
git-tree-sha1 = "bce26c3dab336582805503bed209faab1c279768"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.4"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "6c72198e6a101cccdd4c9731d3985e904ba26037"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3bc002af51045ca3b47d2e1787d6ce02e68b943a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.122"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "c249d86e97a7e8398ce2068dce4c078a1c3464de"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.7.16"

    [deps.DomainSets.extensions]
    DomainSetsMakieExt = "Makie"
    DomainSetsRandomExt = "Random"

    [deps.DomainSets.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.DynamicPolynomials]]
deps = ["Future", "LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Reexport", "Test"]
git-tree-sha1 = "3f50fa86c968fc1a9e006c07b6bc40ccbb1b704d"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.6.4"

[[deps.EnumX]]
git-tree-sha1 = "bddad79635af6aec424f53ed8aad5d7555dc6f00"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.5"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.ExproniconLite]]
git-tree-sha1 = "c13f0b150373771b0fdc1713c97860f8df12e6c2"
uuid = "55351af7-c7e9-48d6-89ff-24e801d99491"
version = "0.10.14"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "173e4d8f14230a7523ae11b9a3fa9edb3e0efd78"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.14.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "4c1acff2dc6b6967e7e750633c50bc3b8d83e617"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.3"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IntervalSets]]
git-tree-sha1 = "5fbb102dcb8b1a858111ae81d56682376130517d"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.11"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.Jieko]]
deps = ["ExproniconLite"]
git-tree-sha1 = "2f05ed29618da60c06a87e9c033982d4f71d0b6c"
uuid = "ae98c720-c025-4a4a-838c-29b094483192"
version = "0.2.1"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4255f0032eafd6451d707a51d5f0248b8a165e4d"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.3+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.Moshi]]
deps = ["ExproniconLite", "Jieko"]
git-tree-sha1 = "53f817d3e84537d84545e0ad749e483412dd6b2a"
uuid = "2e0e35c7-a2e4-4343-998d-7ef72827ed2d"
version = "0.3.7"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.5.20"

[[deps.MultivariatePolynomials]]
deps = ["DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "d38b8653b1cdfac5a7da3b819c0a8d6024f9a18c"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.5.13"
weakdeps = ["ChainRulesCore"]

    [deps.MultivariatePolynomials.extensions]
    MultivariatePolynomialsChainRulesCoreExt = "ChainRulesCore"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "22df8573f8e7c593ac205455ca088989d0a2c7a0"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.6.7"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f07c06228a1c670ae4c87d1276b92c7c597fdda0"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.35"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "f53232a27a8c1c836d3998ae1e17d898d4df2a46"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.72"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "PrecompileTools"]
git-tree-sha1 = "c05b4c6325262152483a1ecb6c69846d2e01727b"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "0.4.34"

    [deps.PreallocationTools.extensions]
    PreallocationToolsForwardDiffExt = "ForwardDiff"
    PreallocationToolsReverseDiffExt = "ReverseDiff"
    PreallocationToolsSparseConnectivityTracerExt = "SparseConnectivityTracer"

    [deps.PreallocationTools.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface"]
git-tree-sha1 = "51bdb23afaaa551f923a0e990f7c44a4451a26f1"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.39.0"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsKernelAbstractionsExt = "KernelAbstractions"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsSparseArraysExt = ["SparseArrays"]
    RecursiveArrayToolsStructArraysExt = "StructArrays"
    RecursiveArrayToolsTablesExt = ["Tables"]
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "52b99504e2c174d9a8592a89647f5187063d1eb1"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.2"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "86a8a8b783481e1ea6b9c91dd949cb32191f8ab4"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.15"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "Adapt", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "Moshi", "PreallocationTools", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLPublic", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface"]
git-tree-sha1 = "7680fbbc8a4fdf9837b4cae5e3fbebe53ec8e4ff"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.122.0"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseDistributionsExt = "Distributions"
    SciMLBaseEnzymeExt = "Enzyme"
    SciMLBaseForwardDiffExt = "ForwardDiff"
    SciMLBaseMLStyleExt = "MLStyle"
    SciMLBaseMakieExt = "Makie"
    SciMLBaseMeasurementsExt = "Measurements"
    SciMLBaseMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    SciMLBaseMooncakeExt = "Mooncake"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseReverseDiffExt = "ReverseDiff"
    SciMLBaseTrackerExt = "Tracker"
    SciMLBaseZygoteExt = ["Zygote", "ChainRulesCore"]

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    MLStyle = "d8e11817-5142-5d16-987a-aa16d5891078"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLOperators]]
deps = ["Accessors", "ArrayInterface", "DocStringExtensions", "LinearAlgebra", "MacroTools"]
git-tree-sha1 = "c1053ba68ede9e4005fc925dd4e8723fcd96eef8"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "1.9.0"
weakdeps = ["SparseArrays", "StaticArraysCore"]

    [deps.SciMLOperators.extensions]
    SciMLOperatorsSparseArraysExt = "SparseArrays"
    SciMLOperatorsStaticArraysCoreExt = "StaticArraysCore"

[[deps.SciMLPublic]]
git-tree-sha1 = "ed647f161e8b3f2973f24979ec074e8d084f1bee"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.0.0"

[[deps.SciMLStructures]]
deps = ["ArrayInterface"]
git-tree-sha1 = "566c4ed301ccb2a44cbd5a27da5f885e0ed1d5df"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f2685b435df2613e25fc10ad8c26dddb8640f547"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.6.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "b8693004b385c842357406e3af647701fe783f98"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.15"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "a136f98cefaf3e2924a66bd75173d1c891ab7453"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.7"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "1ec049c79e13fb2638ddbf8793ab2cbbeb266f45"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.1"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "94c58884e013efff548002e8dc2fdd1cb74dfce5"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.46"

    [deps.SymbolicIndexingInterface.extensions]
    SymbolicIndexingInterfacePrettyTablesExt = "PrettyTables"

    [deps.SymbolicIndexingInterface.weakdeps]
    PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"

[[deps.SymbolicLimits]]
deps = ["SymbolicUtils"]
git-tree-sha1 = "f75c7deb7e11eea72d2c1ea31b24070b713ba061"
uuid = "19f23fe9-fdab-4a78-91af-e7b7767979c3"
version = "0.2.3"

[[deps.SymbolicUtils]]
deps = ["AbstractTrees", "ArrayInterface", "Bijections", "ChainRulesCore", "Combinatorics", "ConstructionBase", "DataStructures", "DocStringExtensions", "DynamicPolynomials", "ExproniconLite", "LinearAlgebra", "MultivariatePolynomials", "NaNMath", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicIndexingInterface", "TaskLocalValues", "TermInterface", "TimerOutputs", "Unityper"]
git-tree-sha1 = "a85b4262a55dbd1af39bb6facf621d79ca6a322d"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "3.32.0"

    [deps.SymbolicUtils.extensions]
    SymbolicUtilsLabelledArraysExt = "LabelledArrays"
    SymbolicUtilsReverseDiffExt = "ReverseDiff"

    [deps.SymbolicUtils.weakdeps]
    LabelledArrays = "2ee39098-c373-598a-b85f-a56591580800"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

[[deps.Symbolics]]
deps = ["ADTypes", "ArrayInterface", "Bijections", "CommonWorldInvalidations", "ConstructionBase", "DataStructures", "DiffRules", "Distributions", "DocStringExtensions", "DomainSets", "DynamicPolynomials", "LaTeXStrings", "Latexify", "Libdl", "LinearAlgebra", "LogExpFunctions", "MacroTools", "Markdown", "NaNMath", "OffsetArrays", "PrecompileTools", "Primes", "RecipesBase", "Reexport", "RuntimeGeneratedFunctions", "SciMLBase", "SciMLPublic", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArraysCore", "SymbolicIndexingInterface", "SymbolicLimits", "SymbolicUtils", "TermInterface"]
git-tree-sha1 = "1b09f5faec5284f505c40e68ba565115e7d48718"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "6.56.0"

    [deps.Symbolics.extensions]
    SymbolicsD3TreesExt = "D3Trees"
    SymbolicsForwardDiffExt = "ForwardDiff"
    SymbolicsGroebnerExt = "Groebner"
    SymbolicsLuxExt = "Lux"
    SymbolicsNemoExt = "Nemo"
    SymbolicsPreallocationToolsExt = ["PreallocationTools", "ForwardDiff"]
    SymbolicsSymPyExt = "SymPy"
    SymbolicsSymPyPythonCallExt = "SymPyPythonCall"

    [deps.Symbolics.weakdeps]
    D3Trees = "e3df1716-f71e-5df9-9e2d-98e193103c45"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Groebner = "0b43b601-686d-58a3-8a1c-6623616c7cd4"
    Lux = "b2108857-7c20-44ae-9111-449ecde12c47"
    Nemo = "2edaba10-b0f1-5616-af89-8c11ac63239a"
    PreallocationTools = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TaskLocalValues]]
git-tree-sha1 = "67e469338d9ce74fc578f7db1736a74d93a49eb8"
uuid = "ed4db957-447d-4319-bfb6-7fa9ae7ecf34"
version = "0.1.3"

[[deps.TermInterface]]
git-tree-sha1 = "d673e0aca9e46a2f63720201f55cc7b3e7169b16"
uuid = "8ea1fca8-c5ef-4a55-8b96-4e9afe9c9a3c"
version = "2.0.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "3748bd928e68c7c346b52125cf41fff0de6937d0"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.29"

    [deps.TimerOutputs.extensions]
    FlameGraphsExt = "FlameGraphs"

    [deps.TimerOutputs.weakdeps]
    FlameGraphs = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"

[[deps.Tricks]]
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Unityper]]
deps = ["ConstructionBase"]
git-tree-sha1 = "25008b734a03736c41e2a7dc314ecb95bd6bbdb0"
uuid = "a7c27f48-0311-42f6-a7f8-2c11e75eb415"
version = "0.1.6"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╟─de19ba50-af83-11f0-8445-075b7cd7e011
# ╟─1d9e9278-4dcd-4a22-a3ff-2a0d5426e4c1
# ╠═74681217-659d-49d8-aa1e-c0c863ea2144
# ╠═0a7ea06a-3e3c-400c-8449-dca4e8ae0373
# ╟─e24c29c0-38f1-48c4-9811-4378fa4adcbe
# ╟─d2e065a5-14b1-4558-a7d2-8c3d777c6c5d
# ╠═7c7d4480-de38-4f5e-9a73-e43c77c29881
# ╟─82ca69a8-aa6d-4f90-a912-b2f848e190a3
# ╠═d6712129-dcce-4884-8bb2-12724fff2622
# ╠═e86e42cf-5787-452e-992a-4db9a48f2bd1
# ╟─27cb8eac-a052-45b6-9b4e-55ee0dd72048
# ╟─2eb2e25c-df4b-4059-8395-c1fab6400bdf
# ╠═b8ca321b-736c-4a29-bec6-041bdb4c28fc
# ╟─03ddb110-d167-427a-9694-06dd09a59735
# ╠═4dce8bce-42b1-4637-a486-f7c6692664bf
# ╠═d10c042c-0147-4288-9eba-abc4d35341ff
# ╠═dab41bab-f9f1-4b04-bcae-f2d1a8d6bcc0
# ╠═54e9a1df-420d-496f-bcd0-eb1448edcce6
# ╠═e406d151-214e-4f40-9623-29bb636ef871
# ╟─8787daf9-52a1-469f-bd9d-aba957c67318
# ╟─2c41d052-6f0a-40af-be12-6f193b24b971
# ╟─7aaeb842-b7bd-4081-8a46-db7b0f3459b0
# ╠═cac73c9a-0a18-4c1d-87eb-b54bb991415b
# ╟─5a590e7e-7837-4a37-99bd-c56370fc41f2
# ╠═01c7ee06-b5ed-48aa-838d-1e2635edae45
# ╟─162800f3-c354-4ea6-8259-aabe1605178c
# ╠═81ed17a9-d183-47b0-a4c7-593346ebf05e
# ╟─4130c505-edb5-41af-bcce-5fa8181763a3
# ╠═42f832ad-82b3-4076-ae69-6bf14ead9038
# ╠═24b9af40-f2c0-48f1-a442-24a7dfdd1ebf
# ╟─0e264270-8dff-4359-bb37-be5578bd7c33
# ╠═a636e3e2-9f63-48ce-94dd-3006bafab90e
# ╠═01eb43eb-90bf-4b9f-9f77-9dc2015be8b0
# ╟─f12b46af-0d0c-458a-897d-66eeab419ece
# ╟─854e677c-bd3f-4e0f-9e07-6818a8713411
# ╠═9cf8876a-0feb-4ace-92f0-4a5ee9753729
# ╟─bea2fffb-e2d6-4f34-8a0a-9d632d5a12bd
# ╠═c3c7fd11-7f6e-48ad-bbbd-822bf3665248
# ╠═b1598942-2300-4a3c-bb08-1f6dadcbd4c6
# ╠═97206602-e5fe-4c7e-bbbc-1a68daa46dfc
# ╟─58232802-7f9e-4566-96af-14834f8ab885
# ╠═61826d69-d8c0-4861-9239-a474a49f18f2
# ╠═6d2f263a-5d86-41f8-8d7c-2cc761cb0555
# ╠═390f48c9-5f46-4a2e-8796-942f2cc7ccc8
# ╠═34789604-2744-4f19-aced-aa7e6e74d689
# ╟─8c01d6c6-3b89-4539-a78f-9965de105c58
# ╠═58c510a6-a976-454b-98d5-f56f1b0e0b6a
# ╠═8eb80e66-2a15-466c-87ed-4605469a35db
# ╟─9d2cd682-119f-47a4-8af5-b74433b7dca2
# ╠═46fa536a-635e-48e7-9401-d3cf067a5481
# ╠═8aa901d5-d29a-48b3-934d-492222834988
# ╟─ed8ec309-c5df-47b1-8172-a3040af93cd8
# ╠═e3a7decc-848d-4581-9a62-94a003b3af9b
# ╟─7323f5ea-2873-4fe9-9c41-8ad095af807d
# ╠═a62415df-7864-4b1b-859a-51c763fdbbfd
# ╠═0b1cee5d-3cdd-41b2-bd12-206759d07dc1
# ╠═4be13c18-144e-4cc9-a84e-35c9f2b247af
# ╟─2b005fd0-fa15-490a-8e3c-d8bbc38fce27
# ╠═e17d8292-1394-4d14-9d83-0cbb1dc02ce8
# ╠═84812014-4448-4639-a321-05f97798a860
# ╠═3f0cc04a-d82f-4ac0-b8a5-e4d2a7fe0c3c
# ╠═095aab17-491b-48d9-8124-a92d737bd683
# ╠═f76836b6-dced-420c-b971-6bd1f03f7942
# ╠═dc9e185a-796c-43ef-9a25-7f2a84d9f69a
# ╠═0bedb30e-27c2-4552-bb02-9585ac64facf
# ╠═2c2208ab-b29c-4859-818a-42a7b2f1e62c
# ╠═c1b87c8c-73f3-4cce-8bf1-11028b30f2c2
# ╠═41408145-200f-4b36-b631-fc6b8bffcce4
# ╠═47e54957-265c-4964-b1c2-53c55c40195c
# ╟─f5d16cbe-5daa-4273-9e13-f8ba45f8bf20
# ╠═f0190ecf-47a0-4618-a8f6-c8a3172a49ad
# ╠═4675c4cf-a6f9-4109-8d00-722a648acb5d
# ╠═03cc182f-6cdc-4d40-ae30-10704bcb440c
# ╠═99462628-168a-4cc5-a1a0-ed5d5244d509
# ╠═e7f1a6a4-c445-4514-a972-e2fe0e9e1d0f
# ╠═8252153d-2a3c-4615-be65-f23e57dda841
# ╟─4cb1b685-f801-420b-bd70-c1f4c771d0bf
# ╠═2a4977ab-9749-49f8-a844-3791e015ac7c
# ╟─cac92ee9-ee5d-4ded-8760-cf593f2eb342
# ╠═3885fa29-592a-4309-9147-9d726078cd9a
# ╠═707cb394-f408-4088-a76e-57811d12002d
# ╟─a9a6f108-80a3-4f30-b754-36c4c06a7bd2
# ╠═1aba3e89-98e6-42d6-9fdc-1256e99898f7
# ╠═bbdc708e-293d-41da-b756-c28d03da81fe
# ╠═b13c0fd9-009e-44be-a328-06ab432a6b33
# ╠═36f0e627-d25b-4ab8-8968-b0bf9775a2fa
# ╟─afb53e8a-e975-40e2-b663-5b71e9a8e865
# ╠═2a44be5c-5454-4d3c-9338-321d28f251e6
# ╠═5bce83ec-ce0e-447a-9b4b-d44db39a6493
# ╠═fc4d4dae-afa4-49d3-bbde-78ec41847001
# ╟─a6175368-8212-42c5-8e32-a243ae23019d
# ╠═4fe33ed1-a20c-4840-9ab4-42828e7b8804
# ╟─ee3d0cec-08e1-4408-a995-a1a6198e608b
# ╠═e25b9ee5-bacb-42e1-a947-3dcc8d158368
# ╟─c788f25b-0f77-4396-a999-9917f99a872f
# ╠═57fdca80-e35f-40bc-8ba0-83d1d777dcbc
# ╠═eb13a1cd-7769-4fe2-a8df-e800c34ce7c1
# ╠═be38cb8c-2825-4738-b490-5e2a89398cc7
# ╠═cca392b5-24c0-4899-af35-972d1ca6777e
# ╟─5de22a47-1386-4b43-b087-224339148166
# ╟─54ca2866-7cea-4c90-978c-e7a1031fc6ec
# ╠═8ccdf71e-3c2f-45fa-afb0-67dadeba12da
# ╟─731769df-3a74-41b3-81a8-1ae226d4bb29
# ╟─6cd9539e-c4e0-4333-af8f-292e47cc539b
# ╟─4d436b91-06dc-462b-a2e2-f8704d50d37a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
