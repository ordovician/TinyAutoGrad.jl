using TinyAutoGrad
using Test

@testset "TinyAutoGrad.jl" begin
    @testset "Simple Calculations" begin
        x = Dual(3)
        y = Dual(6)
    
        @test x^2 == Dual(9, 6) # Because dx^2/dx = 2x
        @test y^2 == Dual(36, 12)
    end
    
    @testset "Derivative" begin
        f(x) = x^2
        g(x) = 6x^3 + 2x^2
        df(x) = derive(f, x)
        dg(x) = derive(g, x)
        
        # Manually derive g(x) for comparison
        h(x) = 18x^2 + 4x
        
        @test df(3) == 2*3 # Because f'(x) = 2x
        @test dg(1) == h(1)
        @test dg(2) == h(2)
    end
    
    @testset "Gradients" begin
        f(x, y) = 3x + 10y
        g(x, y) = 3x^2 + 10y^2
        
        @test gradient(f, 1, 1) == [3, 10]
        @test gradient(f, 2, 3) == [3, 10]
        
        @test gradient(g, 1, 1) == [6, 20]
        @test gradient(g, 2, 3) == [12, 60]
    end
    
    @testset "Gradient of Params" begin
        f(x, y) = 3x + 10y
        
        ps = Params()
        
        g() = 3ps.x + 10ps.y
        
        ∇f = gradient(f, 1, 1)
        ps.x = 1
        ps.y = 1
        ∇g = gradient(g, ps)
        
        @test ∇f[1] == ∇g.x
        @test ∇f[2] == ∇g.y
                
        ∇f = gradient(f, 2, 3)
        ps.x = 2
        ps.y = 3
        ∇g = gradient(g, ps)
        
        @test ∇f[1] == ∇g.x
        @test ∇f[2] == ∇g.y
    end

end
