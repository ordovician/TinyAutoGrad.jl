# Read an explanation of this implementation here:
#   https://nextjournal.com/erik-engheim/implementation-of-automatic-differentiation
export Dual, derive, gradient, Params

import Base: +, -, *, /, sin, cos, show,
             convert, promote_rule,
             setproperty!, getproperty


struct Dual{T<:Real} <: Real
    x::T
    ϵ::T
end

"""
    Dual(x)

Create a dual number of `x`. Dual numbers allow us to keep track
of derivatives when performing calculations. If `f(x) = x` then `f'(x) = 1`.
That is why `Dual(x)` is equal to `Dual(x, 1)`. 
"""
Dual(x) = Dual(x, one(x))

show(io::IO, d::Dual) = print(io, d.x, " + ", d.ϵ, "ϵ")


a::Dual + b::Dual = Dual(a.x + b.x, a.ϵ + b.ϵ)
a::Dual - b::Dual = Dual(a.x - b.x, a.ϵ - b.ϵ)


a::Dual * b::Dual = Dual(a.x * b.x, b.x * a.ϵ + a.x * b.ϵ)
a::Dual / b::Dual = Dual(a.x / b.x, (a.ϵ*b.x - b.ϵ*a.x) / b.x^2)

sin(d::Dual) = Dual(sin(d.x), cos(d.x) * d.ϵ)
cos(d::Dual) = Dual(cos(d.x), sin(d.x) * (-d.ϵ))

function promote_rule(::Type{Dual{T}}, ::Type{Dual{R}}) where {T,R} 
    Dual{promote_type(T,R)}
end

function promote_rule(::Type{Dual{T}}, ::Type{R}) where {T,R}
    Dual{promote_type(T,R)}
end

function convert(::Type{Dual{T}}, x::Dual) where T
    Dual(convert(T, x.x), convert(T, x.ϵ))
end

function convert(::Type{Dual{T}}, x::Real) where T
    Dual(convert(T, x), zero(T))
end

"""
    derive(f, x)

Given a function `y = f(x)` then `y` is the dependent variable and `x` is
the independent variable. `derive(f, x)` calculates the derivative of the function `f`
with respect to the indepedent variable `x`.
"""
derive(f, x) = f(Dual(x)).ϵ

"""
    gradient(f, xs...)

Calculate the gradient of the function `f` with respect to the variables `xs`.
So say we got `xs = [x₁, x₂, ..., xₙ]`. Then the gradient `gs = ∇f(x₁, x₂, ..., xₙ)`
is such that `gs[1] = ∂f(xs...)/∂x₁`, `gs[2] = ∂f(xs...)/∂x₂` etc. So the component `gs[i]`
of the derivative is partial derivative of `f` with respect to `xs[i]`.
"""
function gradient(f, xs...)
    gs = []
    for (i, x) in enumerate(xs)
        if i == 1
            push!(gs, f(Dual(x), xs[2:end]...))
        else
            push!(gs, f(xs[1:(i-1)]..., Dual(x), xs[(i+1):end]...))
        end
    end
    [g.ϵ for g in gs]
end


const Params = Dict{Symbol, Real}

function getproperty(params::Params, key::Symbol)
    if hasfield(Params, key)
        getfield(params, key)
    else
        getindex(params, key)
    end
end

function setproperty!(params::Params, key::Symbol, x::Real)
    if hasfield(Params, key)
        setfield!(params, key, x)
    else
        setindex!(params, x, key)
    end
end

"""
    gradient(f, params::Params)

Calculate the gradient of the function `f` with respect to the parameters `params`.
"""
function gradient(f, params::Params)
    gs = Params()       # To hold gradient

    for k in keys(params)
        backup = params[k]
        params[k] = Dual(backup) # So we can derivate with respect to paramer `k`
        gs[k] = f().ϵ
        params[k] = backup
    end

    gs
end
